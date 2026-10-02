MTP_SHARED_EMBD_ARCHS = frozenset({"qwen4exp"})


def validate_mtp_shared_embd(model_arch: str) -> None:
    if model_arch not in MTP_SHARED_EMBD_ARCHS:
        raise ValueError(
            f"--mtp-shared-embd is not supported for {model_arch}; shared-weight borrowing is only implemented for qwen4exp"
        )
