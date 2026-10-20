use super::*;

pub(super) fn delay(value: &Array) -> Result<Array> {
    let outputs = crate::array::metal_kernel(
        "mlxl3_test_slow_resident_experts",
        &["value"],
        &["out", "stamp"],
        &format!("#define COUNT {}u\n", value.shape().iter().product::<i32>()),
        "uint i=thread_position_in_grid.x; uint v=i+1u; for(uint j=0;j<4000000u;++j) { v^=v<<13; v^=v>>17; v^=v<<5; } stamp[i]=v; for(uint k=i;k<COUNT;k+=512u) out[k]=value[k];",
        &[value],
        &[value.shape().to_vec(), vec![512]],
        &[Dtype::Float16, Dtype::UInt32],
        [512, 1, 1],
        [128, 1, 1],
    )?;
    Array::async_eval_all(&[&outputs[0], &outputs[1]])?;
    outputs[0].try_clone()
}

fn prime_half(moe: &mut SsdMoe, input: &Array) -> Result<Vec<u32>> {
    let ids = moe.route(input)?.0.to_u32()?;
    ensure!(ids.len() == 10 && ids.iter().copied().collect::<BTreeSet<_>>().len() == 10);
    let routes = moe.cache.borrow_mut().reserve(&ids[..5])?;
    let jobs = ids[..5]
        .iter()
        .zip(routes)
        .map(|(&id, (slot, _))| (id, slot))
        .collect::<Vec<_>>();
    moe.fill_many(&jobs)?;
    Ok(ids)
}

#[test]
#[ignore = "physical Apple GPU; verify early hot does not block routes or demanded reads"]
fn demanded_reads_start_while_gpu_resident_experts_are_pending() -> Result<()> {
    let (checkpoint, input, expected) = super::tests::admission_fixture()?;
    let mut moe = SsdMoe::load(&checkpoint, "model.language_model.layers.0.mlp", 24, true)?;
    moe.gpu_hot = true;
    prime_half(&mut moe, &input)?;
    moe.slow_hot_for_test = true;
    let actual = moe.forward(&input)?;
    println!(
        "SSD_GPU_HOT_SCHEDULE {}",
        serde_json::json!({
            "submissions":moe.hybrid_submissions,"ready_at_routes":moe.hot_ready_at_routes,
            "ready_at_demand":moe.hot_ready_at_demand,"events":moe.schedule_events,
            "read_bytes":moe.read_bytes,"hits":moe.hits,"misses":moe.misses,
        })
    );
    ensure!(actual.shape() == [1, H] && actual.to_bytes()? == expected);
    ensure!(moe.hybrid_submissions == 1 && moe.read_bytes == 10 * EXPERT_BYTES);
    ensure!(moe.hits == 5 && moe.misses == 5);
    ensure!(
        moe.hot_ready_at_routes == Some(false),
        "routes waited for resident compute"
    );
    ensure!(
        moe.hot_ready_at_demand == Some(false),
        "demand waited for resident compute"
    );
    moe.drain()?;
    Ok(())
}

#[test]
#[ignore = "physical Apple GPU; error before reservation preserves only already filled residents"]
fn early_hot_failure_preserves_ready_keys_and_recovers() -> Result<()> {
    let (checkpoint, input, expected) = super::tests::admission_fixture()?;
    let mut moe = SsdMoe::load(&checkpoint, "model.language_model.layers.0.mlp", 24, true)?;
    moe.gpu_hot = true;
    let ids = prime_half(&mut moe, &input)?;
    let invalid = moe.banks[0].reshape(&[24, -1])?;
    let bank = std::mem::replace(&mut moe.banks[0], invalid);
    let error = moe
        .forward(&input)
        .err()
        .context("invalid hot bank succeeded")?;
    moe.banks[0] = bank;
    ensure!(
        error
            .to_string()
            .contains("expert slots were validated for a different bank capacity"),
        "{error:#}"
    );
    ensure!(moe.read_bytes == 5 * EXPERT_BYTES && moe.misses == 0 && moe.hits == 0);
    ensure!(ids[..5].iter().all(|&id| moe.cache.borrow().contains(id)));
    ensure!(ids[5..].iter().all(|&id| !moe.cache.borrow().contains(id)));
    moe.drain()?;
    let actual = moe.forward(&input)?;
    ensure!(actual.shape() == [1, H] && actual.to_bytes()? == expected);
    ensure!(moe.read_bytes == 10 * EXPERT_BYTES && moe.misses == 5);
    moe.drain()?;
    println!("SSD_GPU_HOT_EARLY_FAILURE parity=byte_exact ready_keys=5 recovery=true");
    Ok(())
}

#[cfg(kani)]
#[kani::proof]
fn actual_hybrid_hot_demand_entry() {
    demanded_reads_start_while_gpu_resident_experts_are_pending().unwrap();
}
