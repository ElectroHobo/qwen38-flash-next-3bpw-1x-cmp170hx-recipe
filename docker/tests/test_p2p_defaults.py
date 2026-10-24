from pathlib import Path
ROOT = Path(__file__).parents[2]
def test_one_card_samples_turn_peer_access_off():
    # One card has no peer traffic: the samples say off and serve.sh forces it.
    assert "\nP2P=off\n" in (ROOT / "docker/container.env").read_text()
    assert "\nVLLM_CUSTOM_ALLREDUCE_PCIE_P2P=0\n" not in (ROOT / "docker/container.env").read_text()
    assert "P2P=" not in (ROOT / ".env.example").read_text()
    serve = (ROOT / "serve.sh").read_text()
    assert "VLLM_CUSTOM_ALLREDUCE_PCIE_P2P:-0" not in serve
    assert "QWEN_DEFAULT_PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False" in serve
    assert "PYTORCH_CUDA_ALLOC_CONF-$QWEN_DEFAULT_PYTORCH_CUDA_ALLOC_CONF" in serve
