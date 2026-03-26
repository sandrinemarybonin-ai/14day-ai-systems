import os
from typing import Dict, Any

from src.workflow_steps import (
    step1_load_input,
    step2_extract_structured,
    step3_classify_and_route,
    step4_generate_draft_reply,
    step5_save_outputs,
    step6_log_run
)


def run_workflow(input_file: str, out_dir: str, log_file: str) -> Dict[str, Any]:
    s1 = step1_load_input(input_file)
    if not s1["ok"]:
        step6_log_run(log_file, {"ok": False, "step": 1, "error": s1.get("error")})
        return s1

    s2 = step2_extract_structured(s1["raw_text"])
    if not s2["ok"]:
        step6_log_run(log_file, {"ok": False, "step": 2, "error": s2.get("error")})
        return s2

    extracted = s2["extracted"]

    s3 = step3_classify_and_route(extracted)

    s4 = step4_generate_draft_reply(extracted, s3["route"], s3["sla"])

    payload = {
        "input_file": input_file,
        "extracted": extracted,
        "route": s3["route"],
        "sla": s3["sla"],
        "draft_reply": s4["draft_reply"],
    }

    out_base = os.path.join(out_dir, os.path.splitext(os.path.basename(input_file))[0])
    s5 = step5_save_outputs(out_base, payload)

    step6_log_run(log_file, {"ok": True, "file": input_file})

    return {"ok": True, "out_base": out_base, "saved": s5}