//! Native Qwen3.5 MoE feed-forward block, used to validate the model port layerwise.
use crate::{array::Array, checkpoint::Checkpoint, lfm2::Projection, moe::Exl3SwitchGlu, router};
use anyhow::{Result, ensure};

pub struct Moe {
    gate: Projection,
    experts: Exl3SwitchGlu,
    shared_gate: Projection,
    shared_up: Projection,
    shared_down: Projection,
    shared_multiplier: Projection,
    hidden: i32,
    top_k: usize,
}

impl Moe {
    pub fn load(
        checkpoint: &Checkpoint,
        prefix: &str,
        hidden: i32,
        expert_hidden: i32,
        num_experts: i32,
        top_k: usize,
    ) -> Result<Self> {
        ensure!(
            top_k > 0 && top_k <= num_experts as usize,
            "invalid MoE top-k"
        );
        Ok(Self {
            gate: Projection::load(
                checkpoint,
                &format!("{prefix}.gate"),
                hidden,
                num_experts,
                false,
            )?,
            experts: Exl3SwitchGlu::from_checkpoint(
                checkpoint,
                &format!("{prefix}.experts"),
                num_experts,
                top_k as i32,
            )?,
            shared_gate: Projection::load(
                checkpoint,
                &format!("{prefix}.shared_expert.gate_proj"),
                hidden,
                expert_hidden,
                false,
            )?,
            shared_up: Projection::load(
                checkpoint,
                &format!("{prefix}.shared_expert.up_proj"),
                hidden,
                expert_hidden,
                false,
            )?,
            shared_down: Projection::load(
                checkpoint,
                &format!("{prefix}.shared_expert.down_proj"),
                expert_hidden,
                hidden,
                false,
            )?,
            shared_multiplier: Projection::load(
                checkpoint,
                &format!("{prefix}.shared_expert_gate"),
                hidden,
                1,
                false,
            )?,
            hidden,
            top_k,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        ensure!(x.shape() == [1, self.hidden], "Qwen MoE expects one token");
        let probabilities = self.gate.forward(x)?.softmax_precise()?;
        let (selected, scores) = router::topk(&probabilities, self.top_k, true)?;
        let routed = self.experts.forward(x, &selected, &scores)?;
        let shared = self.shared_down.forward(
            &self
                .shared_gate
                .forward(x)?
                .swiglu(&self.shared_up.forward(x)?)?,
        )?;
        routed.add(&shared.mul(&self.shared_multiplier.forward(x)?.sigmoid()?)?)
    }
}
