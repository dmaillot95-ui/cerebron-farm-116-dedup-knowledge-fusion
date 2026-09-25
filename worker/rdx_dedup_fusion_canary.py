import hashlib,json,pathlib,re,sys,unicodedata
records=[
 {"id":"A","content":"Cold benchmarks must never enter training data.","sources":["SRC-1"]},
 {"id":"B","content":"  cold benchmarks MUST never enter training data! ","sources":["SRC-2"]},
 {"id":"C","content":"Retrieval memory is not equivalent to neural training.","sources":["SRC-3"]},
 {"id":"D","content":"Retrieval memory is not equivalent to neural training.","sources":["SRC-4","SRC-5"]}
]
def norm(s):
    s=unicodedata.normalize("NFKC",s).lower()
    s=re.sub(r"[^a-z0-9]+"," ",s)
    return " ".join(s.split())
groups={}
for r in records:
    fp=hashlib.sha256(norm(r["content"]).encode()).hexdigest()
    g=groups.setdefault(fp,{"members":[],"sources":set(),"canonical_content":norm(r["content"])})
    g["members"].append(r["id"]); g["sources"].update(r["sources"])
merged=[{"fingerprint":k,"members":sorted(v["members"]),"sources":sorted(v["sources"]),"canonical_content":v["canonical_content"]} for k,v in groups.items()]
merged=sorted(merged,key=lambda x:x["fingerprint"])
status="PASS" if len(merged)==2 and sorted(len(x["members"]) for x in merged)==[2,2] and sum(len(x["sources"]) for x in merged)==5 else "FAIL"
out={
 "schema":"F116_RDX_DEDUP_FUSION_CANARY_V1","status":status,"farm_id":116,
 "input_count":len(records),"dedup_group_count":len(merged),
 "groups":merged,"training_executed":False,
 "claim_ceiling":"NORMALIZED_EXACT_DEDUP_AND_PROVENANCE_FUSION_CANARY_ONLY_NOT_SEMANTIC_DEDUP"
}
out["receipt_sha256"]=hashlib.sha256(json.dumps(out,sort_keys=True,separators=(",",":")).encode()).hexdigest()
pathlib.Path("artifacts").mkdir(exist_ok=True)
pathlib.Path("artifacts/rdx_dedup_fusion_canary.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,sort_keys=True)); sys.exit(0 if status=="PASS" else 2)
