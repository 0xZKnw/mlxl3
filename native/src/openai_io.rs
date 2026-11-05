//! Bound a stalled HTTP write through Hyper's own buffering layer.
use hyper::server::accept::Accept;
use std::{
    future::Future,
    io,
    pin::Pin,
    task::{Context, Poll},
    time::Duration,
};
use tokio::{
    io::{AsyncRead, AsyncWrite, ReadBuf},
    net::{TcpListener, TcpStream},
    time::{Sleep, sleep},
};

pub(super) struct Incoming {
    listener: TcpListener,
    write_timeout: Duration,
}

impl Incoming {
    pub(super) fn new(listener: std::net::TcpListener, wait: Duration) -> io::Result<Self> {
        Ok(Self {
            listener: TcpListener::from_std(listener)?,
            write_timeout: wait,
        })
    }
}

impl Accept for Incoming {
    type Conn = DeadlineIo<TcpStream>;
    type Error = io::Error;

    fn poll_accept(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<io::Result<Self::Conn>>> {
        let this = self.get_mut();
        this.listener.poll_accept(cx).map(|result| {
            Some(result.map(|(stream, _)| DeadlineIo::new(stream, this.write_timeout)))
        })
    }
}

pub(super) struct DeadlineIo<S> {
    io: S,
    wait: Duration,
    deadline: Option<Pin<Box<Sleep>>>,
}

impl<S> DeadlineIo<S> {
    pub(super) fn new(io: S, wait: Duration) -> Self {
        Self {
            io,
            wait,
            deadline: None,
        }
    }

    fn check_deadline(&mut self, cx: &mut Context<'_>) -> io::Result<()> {
        if self
            .deadline
            .as_mut()
            .is_some_and(|deadline| deadline.as_mut().poll(cx).is_ready())
        {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "HTTP socket write timed out",
            ));
        }
        Ok(())
    }

    fn pending(&mut self, cx: &mut Context<'_>) -> Poll<io::Result<usize>> {
        self.deadline
            .get_or_insert_with(|| Box::pin(sleep(self.wait)));
        match self.check_deadline(cx) {
            Ok(()) => Poll::Pending,
            Err(error) => Poll::Ready(Err(error)),
        }
    }
}

impl<S: AsyncRead + Unpin> AsyncRead for DeadlineIo<S> {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().io).poll_read(cx, buf)
    }
}

impl<S: AsyncWrite + Unpin> AsyncWrite for DeadlineIo<S> {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        if let Err(error) = this.check_deadline(cx) {
            return Poll::Ready(Err(error));
        }
        match Pin::new(&mut this.io).poll_write(cx, buf) {
            Poll::Pending => this.pending(cx),
            ready => ready,
        }
    }

    fn poll_write_vectored(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[io::IoSlice<'_>],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        if let Err(error) = this.check_deadline(cx) {
            return Poll::Ready(Err(error));
        }
        match Pin::new(&mut this.io).poll_write_vectored(cx, buf) {
            Poll::Pending => this.pending(cx),
            ready => ready,
        }
    }

    fn is_write_vectored(&self) -> bool {
        self.io.is_write_vectored()
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        if let Err(error) = this.check_deadline(cx) {
            return Poll::Ready(Err(error));
        }
        match Pin::new(&mut this.io).poll_flush(cx) {
            Poll::Ready(result) => {
                this.deadline = None;
                Poll::Ready(result)
            }
            Poll::Pending => this.pending(cx).map(|result| result.map(|_| ())),
        }
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        if let Err(error) = this.check_deadline(cx) {
            return Poll::Ready(Err(error));
        }
        match Pin::new(&mut this.io).poll_shutdown(cx) {
            Poll::Ready(result) => {
                this.deadline = None;
                Poll::Ready(result)
            }
            Poll::Pending => this.pending(cx).map(|result| result.map(|_| ())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::time::timeout;

    #[tokio::test]
    async fn a_stalled_writer_wakes_without_a_socket_event() {
        let (writer, _reader) = tokio::io::duplex(1);
        let mut writer = DeadlineIo::new(writer, Duration::from_millis(30));
        let error = timeout(Duration::from_secs(1), writer.write_all(b"xx"))
            .await
            .unwrap()
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::TimedOut);
    }

    #[tokio::test]
    async fn partial_progress_does_not_extend_the_write_phase() {
        let (writer, mut reader) = tokio::io::duplex(1);
        let mut writer = DeadlineIo::new(writer, Duration::from_millis(50));
        let consumer = tokio::spawn(async move {
            let mut byte = [0];
            loop {
                tokio::time::sleep(Duration::from_millis(5)).await;
                if reader.read(&mut byte).await.unwrap() == 0 {
                    break;
                }
            }
        });
        let error = timeout(Duration::from_secs(1), writer.write_all(&[b'x'; 1000]))
            .await
            .unwrap()
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::TimedOut);
        drop(writer);
        consumer.await.unwrap();
    }

    #[tokio::test]
    async fn a_completed_flush_allows_a_later_write_phase() {
        let (writer, mut reader) = tokio::io::duplex(1);
        let mut writer = DeadlineIo::new(writer, Duration::from_millis(100));
        let consumer = tokio::spawn(async move {
            let mut bytes = [0; 3];
            tokio::time::sleep(Duration::from_millis(5)).await;
            reader.read_exact(&mut bytes).await.unwrap();
            bytes
        });
        writer.write_all(b"ab").await.unwrap();
        writer.flush().await.unwrap();
        tokio::time::sleep(Duration::from_millis(120)).await;
        writer.write_all(b"c").await.unwrap();
        writer.flush().await.unwrap();
        assert_eq!(consumer.await.unwrap(), *b"abc");
    }
}
