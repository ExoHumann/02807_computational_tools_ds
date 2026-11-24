"""Live FMA Medium MuQ encoder, multi-probe LSH and distinct-track decision."""
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from afp.embed import MuQBackbone
from afp.fingerprint import load_head
from afp.lsh import AngularLSH

ROOT = Path(__file__).resolve().parent.parent
MEDIUM = ROOT/"data/medium"
REVISION = "0562a57814f6f8bbd9fdea0a25921a2fce1a841a"


class Recognizer:
    """Keep the encoder and fingerprints resident on CUDA between uploads."""

    def __init__(self):
        """Load the released Medium artifacts and build the pinned 64-table bucket index."""
        directory = MEDIUM/"data/processed"
        model = os.getenv("MUQ_MODEL_DIR", str(MEDIUM/"hf-cache/hub/models--OpenMuQ--MuQ-large-msd-iter/snapshots"/REVISION))
        precision = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else None
        self.encoder = MuQBackbone(device="cuda", repo=model, autocast_dtype=precision)
        self.head = load_head(MEDIUM/"models/projection_head.pt").to("cuda")
        with np.load(directory/"fingerprints/references.npz") as z:
            self.index = AngularLSH(z["vectors"],z["track_id"],bits=10,tables=64,seed=2807)
        self.vectors = torch.from_numpy(self.index.references).cuda()
        self.metadata = pd.read_csv(directory/"manifest.csv").set_index("track_id")

    @torch.inference_mode()
    def identify(self, payload):
        """Hash one projection and rerank the union of four buckets per table on CUDA."""
        pcm = np.frombuffer(payload,dtype="<i2").astype("f4")[None,:]/32768
        embedding = self.encoder.encode(pcm).astype("f2").astype("f4")
        vector = self.head(torch.from_numpy(embedding).cuda())[0]
        query = vector.cpu().numpy()
        start = perf_counter()
        index = self.index
        projections = (query @ index.planes).reshape(index.tables,index.bits)
        keys = (projections >= 0) @ index.weights
        nearest = np.argsort(abs(projections), axis=1, kind="stable")[:,:2]
        parts = []
        for key, (a,b), (order, offsets) in zip(keys, nearest, index.buckets, strict=True):
            for mask in (0, 1<<a, 1<<b, (1<<a)|(1<<b)):
                k = key ^ mask
                parts.append(order[offsets[k]:offsets[k+1]])
        rows = np.unique(np.concatenate(parts))
        if not rows.size:
            return dict(accepted=False, score=0.,gap=0.,engine="MuQ + LSH · FMA Medium")
        scores = (self.vectors[torch.as_tensor(rows,device="cuda")] @ vector).cpu().numpy()
        best = scores.argmax()
        tid = int(index.track_ids[rows[best]])
        score = float(scores[best])
        other = scores[index.track_ids[rows]!=tid]
        gap = float(score-other.max()) if len(other) else 0.
        elapsed = (perf_counter()-start)*1000
        track = self.metadata.loc[tid]
        return dict(accepted=score>=.749896646 and gap>=.117484212, title=str(track.title), artist=str(track.artist),
                    score=score, gap=gap, track_id=tid,lookup_ms=elapsed,engine="MuQ + LSH · FMA Medium")

