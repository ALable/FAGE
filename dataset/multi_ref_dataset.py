"""K-shot reference frame wrapper for PersonaGazeNet training.

For each (subject, source_frame) pair this wrapper yields an extra
``ref_eye_crops`` tensor of shape ``[K, 3, H_e, W_e*2]`` containing K
distinct reference frames from the same subject. Reference frames are chosen
to have:

    1. head pose close to the source frame  (||Δhead|| ≤ max_head_diff_ref)
    2. *diverse* gaze directions (greedy farthest-point sampling on gaze)
    3. all distinct from each other and from source / target

Falls back gracefully when the user has fewer frames than requested.

This wrapper is purely additive: it does not modify the underlying
``HDFDataset`` so all existing FAGE pipelines keep working.
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch

from dataset.gaze_capture import HDFDataset


class MultiRefHDFDataset(HDFDataset):
    """HDFDataset that also returns K diverse-gaze reference eye crops.

    New keys in the returned dict:
        ref_eye_crops : [K, 3, H_e, W_e*2]   K reference frames (tight crop)
        ref_gazes     : [K, 2]               for diagnostics / curriculum
        ref_heads     : [K, 2]
        ref_K         : int                  actual K returned (≤ requested)
    """

    def __init__(self, *args,
                 ref_K: int = 4,
                 ref_K_choices: List[int] = None,
                 max_head_diff_ref: float = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_K = ref_K
        # Allow randomised K per item (e.g. [1, 4, 8]) — model must be K-agnostic.
        self.ref_K_choices = ref_K_choices
        # If None, defaults to the same head budget as the main pair.
        self.max_head_diff_ref = max_head_diff_ref if max_head_diff_ref is not None else self.max_head_diff

    # ── helpers ───────────────────────────────────────────────────────────────

    def _pick_ref_indices(self, key: str, source_index: int, target_index: int,
                          source_head: np.ndarray, K: int) -> List[int]:
        """Greedy farthest-point sampling on gaze, filtered by head proximity."""
        group = self.hdf[key]
        num_frames = group['pixels'].shape[0]
        if num_frames <= 1:
            return [source_index]

        cand = np.arange(num_frames)
        cand = cand[(cand != source_index) & (cand != target_index)]
        if len(cand) == 0:
            return [source_index]

        labels = group['labels'][cand]                  # [N, 4]
        gazes  = labels[:, 0:2].astype(np.float32)
        heads  = labels[:, 2:4].astype(np.float32)

        if self.max_head_diff_ref is not None:
            head_diff = np.linalg.norm(heads - source_head, axis=1)
            mask = head_diff <= self.max_head_diff_ref
            if mask.any():
                cand   = cand[mask]
                gazes  = gazes[mask]
            # else: keep all (fallback)

        if len(cand) <= K:
            return cand.tolist()

        # Greedy farthest-point sampling in gaze space, seeded from a random sample.
        rng = np.random.RandomState((hash((key, int(source_index))) & 0xFFFFFFFF))
        first = int(rng.randint(len(cand)))
        picked = [first]
        min_dists = np.linalg.norm(gazes - gazes[first], axis=1)
        for _ in range(K - 1):
            nxt = int(np.argmax(min_dists))
            picked.append(nxt)
            new_dist = np.linalg.norm(gazes - gazes[nxt], axis=1)
            min_dists = np.minimum(min_dists, new_dist)
        return [int(cand[i]) for i in picked]

    def _eye_crop_only(self, image_np, face_parsing) -> torch.Tensor:
        """Return only the *tight* (loss-supervision) eye crop, shape [3, H_e, W_e*2]."""
        _, tight, _ = self._crop_eyes(image_np, face_parsing)
        return tight

    # ── public API ────────────────────────────────────────────────────────────

    def __getitem__(self, idx):
        base = super().__getitem__(idx)

        # Resolve the underlying key/index used for the source pair.
        # We can recompute (key, source_index) from idx via index_to_query.
        key, source_index = self.index_to_query[idx]

        # We need target_index too: super() picked it but did not store it.
        # Re-derive deterministically by replaying the picker. To keep cost low
        # we only pull labels (no pixels).
        source_gaze = base['source_gaze'].numpy()
        source_head = base['source_head'].numpy()
        group       = self.hdf[key]
        num_frames  = group['pixels'].shape[0]
        target_index = self._pick_target_index(key, source_index, source_gaze, source_head, num_frames)

        # Pick K reference indices.
        K_req = (int(np.random.choice(self.ref_K_choices))
                 if self.ref_K_choices else self.ref_K)
        ref_idxs = self._pick_ref_indices(key, source_index, target_index, source_head, K_req)
        K_actual = len(ref_idxs)

        ref_crops, ref_gazes, ref_heads = [], [], []
        for ri in ref_idxs:
            img, parsing, gaze, head = self._load_frame(key, ri)
            crop = self._eye_crop_only(img, parsing)
            ref_crops.append(crop)
            ref_gazes.append(torch.from_numpy(gaze))
            ref_heads.append(torch.from_numpy(head))

        # Pad with the last crop if K_actual < requested K — so batches collate.
        while len(ref_crops) < (self.ref_K_choices[-1] if self.ref_K_choices else self.ref_K):
            ref_crops.append(ref_crops[-1].clone())
            ref_gazes.append(ref_gazes[-1].clone())
            ref_heads.append(ref_heads[-1].clone())

        base['ref_eye_crops'] = torch.stack(ref_crops, dim=0)  # [K_max, 3, H_e, W_e*2]
        base['ref_gazes']     = torch.stack(ref_gazes, dim=0)  # [K_max, 2]
        base['ref_heads']     = torch.stack(ref_heads, dim=0)  # [K_max, 2]
        base['ref_K']         = torch.tensor(K_actual, dtype=torch.int32)
        return base
