#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,pathlib,sys,time
import numpy as np
from huggingface_hub import HfApi
from sentence_transformers import SentenceTransformer

MODEL_ID="sentence-transformers/all-MiniLM-L6-v2"

DEV=[
 {"id":"D01","a":"Memory retrieval is not neural learning.","b":"Fetching information from memory does not change neural weights.","label":1},
 {"id":"D02","a":"Cold benchmarks must never enter training.","b":"Sealed evaluation examples have to stay outside training data.","label":1},
 {"id":"D03","a":"Simulation is not physical validation.","b":"A simulated success cannot be claimed as a real-world test.","label":1},
 {"id":"D04","a":"Duplicate outputs are not independent evidence.","b":"Repeated copies from one source do not create separate proof.","label":1},
 {"id":"D05","a":"Memory retrieval is not neural learning.","b":"Neural training modifies model parameters using optimization.","label":0},
 {"id":"D06","a":"Cold benchmarks must never enter training.","b":"Training datasets should record licensing and provenance.","label":0},
 {"id":"D07","a":"Simulation is not physical validation.","b":"Sensitivity analysis ranks influential uncertain parameters.","label":0},
 {"id":"D08","a":"Duplicate outputs are not independent evidence.","b":"A claim should include immutable provenance receipts.","label":0}
]

HOLDOUT=[
 {"id":"H01","a":"Round-trip efficiency includes charging and discharging losses.","b":"Storage efficiency must account for losses on both the charge and discharge legs.","label":1},
 {"id":"H02","a":"Secrets must never enter model training.","b":"Passwords and API credentials are prohibited from training corpora.","label":1},
 {"id":"H03","a":"Content-addressed storage verifies objects using a digest of their bytes.","b":"A cryptographic hash can identify stored content and detect byte changes.","label":1},
 {"id":"H04","a":"On-orbit assembly can bypass fairing size limits.","b":"Joining modules after launch allows structures larger than a launch fairing.","label":1},
 {"id":"H05","a":"Mechanical anchoring can reduce propellant use for service robots.","b":"A robot fixed to the hull need not continuously spend thruster propellant to hold position.","label":1},
 {"id":"H06","a":"Negative results should be retained when verified.","b":"Verified failures are useful evidence because they constrain later claims.","label":1},
 {"id":"H07","a":"Round-trip efficiency includes charging and discharging losses.","b":"Thermal storage moves energy availability across time.","label":0},
 {"id":"H08","a":"Secrets must never enter model training.","b":"Private customer data may be used only with explicit rights and tenant isolation.","label":0},
 {"id":"H09","a":"Content-addressed storage verifies objects using a digest of their bytes.","b":"Semantic vector search ranks documents by embedding similarity.","label":0},
 {"id":"H10","a":"On-orbit assembly can bypass fairing size limits.","b":"Orbital tugs move payloads between staging orbits.","label":0},
 {"id":"H11","a":"Mechanical anchoring can reduce propellant use for service robots.","b":"Manipulator force limits reduce contact damage during assembly.","label":0},
 {"id":"H12","a":"Negative results should be retained when verified.","b":"Independent evidence must not share a duplicated lineage.","label":0}
]

def canon(x):
    return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()

def scores(model,rows):
    a=np.asarray(model.encode([x["a"] for x in rows],normalize_embeddings=True,show_progress_bar=False))
    b=np.asarray(model.encode([x["b"] for x in rows],normalize_embeddings=True,show_progress_bar=False))
    return np.sum(a*b,axis=1)

def select_threshold(dev_scores):
    candidates=sorted(set(float(x) for x in dev_scores))
    if not candidates:
        return 0.5
    cand=[candidates[0]-1e-6]+[(a+b)/2 for a,b in zip(candidates,candidates[1:])]+[candidates[-1]+1e-6]
    best=None
    for t in cand:
        tp=tn=fp=fn=0
        for r,s in zip(DEV,dev_scores):
            p=int(s>=t); y=r["label"]
            tp+=int(p==1 and y==1); tn+=int(p==0 and y==0)
            fp+=int(p==1 and y==0); fn+=int(p==0 and y==1)
        acc=(tp+tn)/len(DEV)
        precision=tp/max(tp+fp,1)
        recall=tp/max(tp+fn,1)
        f1=2*precision*recall/max(precision+recall,1e-12)
        key=(f1,acc,-fp,t)
        if best is None or key>best[0]:
            best=(key,t,{"tp":tp,"tn":tn,"fp":fp,"fn":fn,"accuracy":acc,"f1":f1})
    return best[1],best[2]

def eval_rows(rows,ss,t):
    tp=tn=fp=fn=0; detail=[]
    for r,s in zip(rows,ss):
        p=int(float(s)>=t); y=r["label"]
        tp+=int(p==1 and y==1); tn+=int(p==0 and y==0)
        fp+=int(p==1 and y==0); fn+=int(p==0 and y==1)
        detail.append({"id":r["id"],"label":y,"pred":p,"score":float(s),
                       "a_sha256":hashlib.sha256(r["a"].encode()).hexdigest(),
                       "b_sha256":hashlib.sha256(r["b"].encode()).hexdigest()})
    acc=(tp+tn)/len(rows)
    precision=tp/max(tp+fp,1)
    recall=tp/max(tp+fn,1)
    f1=2*precision*recall/max(precision+recall,1e-12)
    return {"tp":tp,"tn":tn,"fp":fp,"fn":fn,"accuracy":acc,"precision":precision,"recall":recall,"f1":f1},detail

def main():
    t0=time.time()
    sha=HfApi().model_info(MODEL_ID).sha
    model=SentenceTransformer(MODEL_ID,device="cpu")
    ds=scores(model,DEV)
    threshold,dev_metrics=select_threshold(ds)
    hs=scores(model,HOLDOUT)
    hold_metrics,detail=eval_rows(HOLDOUT,hs,threshold)
    thresholds={"holdout_accuracy_min":0.75,"holdout_f1_min":0.75,"false_positive_max":2}
    passed=(hold_metrics["accuracy"]>=thresholds["holdout_accuracy_min"] and
            hold_metrics["f1"]>=thresholds["holdout_f1_min"] and
            hold_metrics["fp"]<=thresholds["false_positive_max"])
    out={
      "schema":"F116_RDX_SEMANTIC_DEDUP_CANARY_V2",
      "status":"PASS" if passed else "HOLD",
      "farm_id":116,
      "role":"DEDUP_KNOWLEDGE_FUSION",
      "model_id":MODEL_ID,
      "resolved_model_revision":sha,
      "method":"COSINE_SIMILARITY_THRESHOLD_SELECTED_ON_DEV_ONLY_THEN_FROZEN_FOR_HOLDOUT",
      "dev_count":len(DEV),"holdout_count":len(HOLDOUT),
      "dev_sha256":hashlib.sha256(canon(DEV)).hexdigest(),
      "holdout_sha256":hashlib.sha256(canon(HOLDOUT)).hexdigest(),
      "selected_threshold":float(threshold),
      "dev_metrics":dev_metrics,
      "holdout_metrics":hold_metrics,
      "acceptance_thresholds":thresholds,
      "holdout_rows":detail,
      "training_executed":False,"weights_changed":False,
      "auto_merge_authorized":False,
      "semantic_candidate_detection_only":True,
      "elapsed_s":round(time.time()-t0,3),
      "claim_ceiling":"SYNTHETIC_SEMANTIC_DUPLICATE_CANDIDATE_DETECTION_CANARY_ONLY_NO_AUTOMATIC_KNOWLEDGE_MERGE"
    }
    out["receipt_sha256"]=hashlib.sha256(canon(out)).hexdigest()
    pathlib.Path("artifacts").mkdir(exist_ok=True)
    pathlib.Path("artifacts/rdx_semantic_dedup_canary_v2.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({"status":out["status"],"selected_threshold":threshold,"dev_metrics":dev_metrics,"holdout_metrics":hold_metrics,"receipt_sha256":out["receipt_sha256"]},sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
