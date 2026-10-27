    #[kani::proof]
    #[kani::unwind(64)]
    fn tool_preparation_tracks_real_stream_filter_transitions() {
        let format: usize = kani::any();
        kani::assume(format < 3);
        let (open, close) = [
            (JSON_OPEN, JSON_CLOSE),
            (GEMMA_OPEN, GEMMA_CLOSE),
            (PYTHON_OPEN, PYTHON_CLOSE),
        ][format];
        let split: usize = kani::any();
        kani::assume(split <= open.len());
        let mut filter = StreamFilter::new();
        assert!(!filter.is_preparing_tool());
        filter.feed(&open[..split]);
        assert_eq!(filter.is_preparing_tool(), split == open.len());
        filter.feed(&open[split..]);
        assert!(filter.is_preparing_tool());
        filter.feed("x");
        assert!(filter.is_preparing_tool());
        filter.feed(close);
        assert!(!filter.is_preparing_tool());
        kani::cover!(split == 0);
        kani::cover!(split == open.len());
        kani::cover!(format == 2 && split > 0 && split < open.len());
    }

