"""
License Plate Detector + Reader — tuned for Indian roads
---------------------------------------------------------
Pipeline: detect plate -> crop -> strip "IND" band -> preprocess -> OCR -> validate -> annotate.

Works on a single image or a video file (frame by frame).

Usage:
    python3 plate_reader.py --input photo.jpg --output result.jpg
    python3 plate_reader.py --input clip.mp4  --output result.mp4

Detection: contour-based (no Haar-cascade XML needed). Looks for rectangular
regions whose aspect ratio matches a typical Indian license plate
(width/height roughly 2.2–7). Falls back to edge+morphology refinement.

OCR: pytesseract (Tesseract OCR), whitelisted to plate characters, with several
page-segmentation modes tried; the best Indian-plate-shaped result is kept.

Indian plate format validated a-posteriori:
    [State][RTO][Series][Number]  e.g. DL01CQ1234, MH12AB3456, KA05EF9012, TN10AA1234
    Regex: ^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{1,4}$
"""

import argparse
import os
import re
import sys

import cv2
import pytesseract

# Whitelist: uppercase letters + digits (Indian plates use these only).
TESSERACT_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Tesseract page-segmentation modes to try (single line, sparse text, etc.).
PSM_MODES = ["7", "6", "11", "13"]

# Indian license-plate regex: state(2) + rto(1-2) + series(1-2) + number(1-4)
INDIAN_PLATE_RE = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{1,4}$")

# Valid Indian state/UT RTO codes.  Used to prefer plates whose first two
# letters are a real state (e.g. KL for Kerala) over a regex-match with an
# impossible state like ZK.
INDIAN_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CH", "CT", "DN", "DD", "DL",
    "GA", "GJ", "HR", "HP", "JH", "JK", "KA", "KL", "LD", "MP",
    "MH", "MN", "ML", "MZ", "NL", "OD", "PB", "PY", "RJ", "SK",
    "TN", "TR", "UP", "UK", "WB", "TS", "LA",
}


class PlateReader:
    def __init__(self):
        pass

    # --------------------------------------------------------------- detect
    def detect(self, frame):
        """
        Contour-based plate detection.

        Strategy:
          1. Grayscale + blackhat morphology to highlight dark text on light
             plates (works for both white private plates and yellow commercial).
          2. Threshold, find contours, approximate polygons.
          3. Keep 4-corner rectangles with plate-like aspect ratio and size.
          4. Non-max suppression to merge overlapping boxes.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Blackhat emphasizes dark regions on light background (plate text).
        blackhat = cv2.morphologyEx(
            gray, cv2.MORPH_BLACKHAT,
            cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        )
        _, thresh = cv2.threshold(
            blackhat, 0, 255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        thresh = cv2.dilate(thresh, None, iterations=1)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        boxes = []
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                x, y, bw, bh = cv2.boundingRect(approx)
                if bw < 70 or bh < 18:
                    continue
                ar = bw / float(bh)
                # Indian plates are wide rectangles; ignore tiny slivers.
                if not (2.2 <= ar <= 7.0):
                    continue
                if bw < w * 0.10 or bh < h * 0.025:
                    continue
                boxes.append((int(x), int(y), int(bw), int(bh)))

        # Also try Canny edges as a secondary net for plates with faint borders.
        boxes += self._detect_canny(frame)
        boxes = self._non_max_suppression(boxes, overlap_thresh=0.45)
        return boxes

    @staticmethod
    def _detect_canny(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        edged = cv2.Canny(blurred, 50, 200)
        edged = cv2.dilate(edged, None, iterations=1)
        contours, _ = cv2.findContours(
            edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        h, w = frame.shape[:2]
        boxes = []
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                x, y, bw, bh = cv2.boundingRect(approx)
                if bw < 70 or bh < 18:
                    continue
                ar = bw / float(bh)
                if 2.2 <= ar <= 7.0 and bw > w * 0.10 and bh > h * 0.025:
                    boxes.append((int(x), int(y), int(bw), int(bh)))
        return boxes

    @staticmethod
    def _non_max_suppression(boxes, overlap_thresh=0.45):
        if not boxes:
            return []
        boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
        picked = []
        for box in boxes:
            x, y, w, h = box
            keep = True
            for px, py, pw, ph in picked:
                ix = max(x, px)
                iy = max(y, py)
                iw = min(x + w, px + pw) - ix
                ih = min(y + h, py + ph) - iy
                if iw > 0 and ih > 0:
                    inter = iw * ih
                    area = min(w * h, pw * ph)
                    if inter / area > overlap_thresh:
                        keep = False
                        break
            if keep:
                picked.append(box)
        return picked

    # ------------------------------------------------------------ preprocess
    @staticmethod
    def _remove_ind_strip(plate_img):
        """
        Indian plates often carry an 'IND' + chakra band on the far left
        (~16-22% of width). Cropping it off prevents it from being read as text.
        """
        _, ww = plate_img.shape[:2]
        cut = int(ww * 0.20)
        return plate_img[:, cut:]

    def preprocess_for_ocr(self, plate_img):
        """Clean up a cropped plate region for Tesseract."""
        plate_img = self._remove_ind_strip(plate_img)
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        hh, ww = gray.shape
        if ww < 320:
            scale = 320 / ww
            gray = cv2.resize(gray, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)

        # CLAHE evens out uneven lighting (night, shadows, glare).
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 12
        )
        return thresh

    # --------------------------------------------------------------- read text
    # Common OCR confusions, applied only where they make structural sense.
    # O->0, I->1, etc. are valid letters in plate series positions (e.g. "CQ"
    # in DL01CQ1234). Blindly translating them globally creates false
    # positives, so we only consider substitutions that keep the overall
    # structure valid.
    _DIGIT_CONFUSABLES = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1",
                          "S": "5", "B": "8", "Z": "2", "G": "6"}
    _LETTER_CONFUSABLES = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z"}

    @classmethod
    def _common_prefix_len(cls, a: str, b: str) -> int:
        n = 0
        for x, y in zip(a, b):
            if x != y:
                break
            n += 1
        return n

    @classmethod
    def _extract_all_matches(cls, text: str):
        """
        Try to extract valid Indian plates from an OCR string.

        Returns a list of (plate_string, edit_type) for every valid match
        found across all edit levels:
          exact > deletion > substitution > double_substitution > substring

        Within each level, all matches are collected (not just the first),
        so the caller can score holistically.
        """
        if not text:
            return []

        def consider(cand: str):
            if INDIAN_PLATE_RE.match(cand):
                return cand
            m = INDIAN_PLATE_RE.search(cand)
            if m:
                return m.group(0)
            return None

        matches = []

        # 1) exact match
        result = consider(text)
        if result:
            matches.append((result, "exact"))

        # 2) single-character deletion (remove noise, keep all real chars)
        for i in range(len(text)):
            result = consider(text[:i] + text[i + 1:])
            if result:
                matches.append((result, "deletion"))

        # 3) single-character substitution in wrong class
        #    (e.g. O read where a digit should be, 0 read where a letter)
        for i, ch in enumerate(text):
            repl_map = (cls._DIGIT_CONFUSABLES if ch.isalpha()
                        else cls._LETTER_CONFUSABLES)
            for repl in repl_map.get(ch, []):
                result = consider(text[:i] + repl + text[i + 1:])
                if result:
                    matches.append((result, "substitution"))

        # 4) double-character substitution (two OCR confusions in one plate)
        n = len(text)
        for i in range(n):
            ch_i = text[i]
            repl_map_i = (cls._DIGIT_CONFUSABLES if ch_i.isalpha()
                          else cls._LETTER_CONFUSABLES)
            for repl_i in repl_map_i.get(ch_i, []):
                text_i = text[:i] + repl_i + text[i + 1:]
                for j in range(i + 1, n):
                    ch_j = text_i[j]
                    repl_map_j = (cls._DIGIT_CONFUSABLES if ch_j.isalpha()
                                  else cls._LETTER_CONFUSABLES)
                    for repl_j in repl_map_j.get(ch_j, []):
                        result = consider(
                            text_i[:j] + repl_j + text_i[j + 1:])
                        if result:
                            matches.append((result, "double_substitution"))

        # 5) substring match (border / IND-strip noise around)
        result = consider(text)
        if result:
            matches.append((result, "substring"))

        return matches

    def read_text(self, plate_img):
        """
        Run Tesseract with several PSM modes; extract the best Indian-plate
        candidate using fuzzy matching (insertion / substitution tolerant).
        """
        processed = self.preprocess_for_ocr(plate_img)
        # Track which PSM modes produced each cleaned candidate; a plate
        # confirmed by multiple PSM modes is more reliable.
        cand_psms: dict[str, set[str]] = {}

        for psm in PSM_MODES:
            config = (f"--oem 3 --psm {psm} "
                      f"-c tessedit_char_whitelist={TESSERACT_WHITELIST}")
            raw = pytesseract.image_to_string(processed, config=config)
            cleaned = re.sub(r"[^A-Z0-9]", "", raw.upper())
            if cleaned:
                cand_psms.setdefault(cleaned, set()).add(psm)

        # Collect every valid plate extraction from every candidate, tracking
        # which PSM modes produced it (more modes = more reliable).
        # Edit cost (fewer edits = more reliable):
        #   exact: 0             (no edits needed)
        #   deletion: 1          (removed one noise char)
        #   substitution: 1      (corrected one misread char)
        #   double_substitution: 2 (two corrections — less reliable)
        #   substring: 3         (may have dropped multiple real chars)
        edit_cost = {"exact": 0, "deletion": 1, "substitution": 1,
                     "double_substitution": 2, "substring": 3}

        # plate -> {"cost": min_edit_cost, "valid_state": bool, "psm_modes": set of PSM modes,
        #           "len": plate length, "prefix": common prefix with source}
        aggregated = {}

        for c, psm_set in cand_psms.items():
            for plate, etype in self._extract_all_matches(c):
                cost = edit_cost[etype]
                valid_state = plate[:2] in INDIAN_STATE_CODES
                prefix = self._common_prefix_len(c, plate)
                if plate not in aggregated:
                    aggregated[plate] = {
                        "cost": cost,
                        "valid_state": valid_state,
                        "psm_modes": set(psm_set),
                        "len": len(plate),
                        "prefix": prefix,
                    }
                else:
                    entry = aggregated[plate]
                    entry["psm_modes"].update(psm_set)
                    entry["cost"] = min(entry["cost"], cost)
                    entry["valid_state"] = entry["valid_state"] or valid_state
                    entry["prefix"] = max(entry["prefix"], prefix)

        if not aggregated:
            # Fallback: longest candidate with enough characters.
            plausible = [c for c in cand_psms if len(c) >= 6]
            if plausible:
                return max(plausible, key=len)
            return ""

        # Pick the best plate:
        #   1. valid Indian state code (e.g. KL for Kerala)
        #   2. most distinct PSM modes that produced it (reliability)
        #   3. fewest edits (lower cost = better)
        #   4. longest plate (preserves more characters)
        #   5. longest common prefix with source (state code usually correct)
        best_plate = max(
            aggregated.keys(),
            key=lambda p: (
                aggregated[p]["valid_state"],
                len(aggregated[p]["psm_modes"]),
                -aggregated[p]["cost"],
                aggregated[p]["len"],
                aggregated[p]["prefix"],
            ),
        )
        return best_plate

    # ------------------------------------------------------------- process
    def process_frame(self, frame):
        """Detect, crop, read, validate, and annotate a single frame.
        Returns (annotated_frame, list of {box, text})."""
        results = []
        boxes = self.detect(frame)
        for (x, y, w, h) in boxes:
            pad_x, pad_y = int(w * 0.06), int(h * 0.18)
            x0, y0 = max(x - pad_x, 0), max(y - pad_y, 0)
            x1, y1 = min(x + w + pad_x, frame.shape[1]), min(y + h + pad_y, frame.shape[0])
            crop = frame[y0:y1, x0:x1]

            text = self.read_text(crop)
            results.append({"box": (x0, y0, x1, y1), "text": text})

            color = (0, 255, 0) if text else (0, 0, 255)
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
            label = text if text else "?"
            label_y = y0 - 10 if y0 - 10 > 10 else y1 + 25
            cv2.putText(
                frame, label, (x0, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA
            )
        return frame, results


# ------------------------------------------------------------------ io helpers
def is_video(path):
    return path.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))


def run_on_image(reader, in_path, out_path):
    frame = cv2.imread(in_path)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {in_path}")
    annotated, results = reader.process_frame(frame)
    cv2.imwrite(out_path, annotated)
    if not results:
        print("No plate detected.")
    for r in results:
        print(f"Plate at {r['box']} -> text: '{r['text'] or '(unreadable)'}'")
    return results


def run_on_video(reader, in_path, out_path):
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {in_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    seen = {}
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        annotated, results = reader.process_frame(frame)
        for r in results:
            if r["text"]:
                seen[r["text"]] = seen.get(r["text"], 0) + 1
        writer.write(annotated)
        frame_idx += 1

    cap.release()
    writer.release()

    print(f"Processed {frame_idx} frames.")
    if seen:
        print("Plate readings (text -> times seen), most frequent first:")
        for text, count in sorted(seen.items(), key=lambda kv: -kv[1]):
            print(f"  {text}: {count}")
    else:
        print("No plate text read in this video.")
    return seen


def main():
    parser = argparse.ArgumentParser(
        description="Detect and read Indian license plates in images/videos."
    )
    parser.add_argument("--input", required=True, help="Path to input image or video")
    parser.add_argument("--output", required=True, help="Path to save annotated output")
    args = parser.parse_args()

    reader = PlateReader()

    if is_video(args.input):
        run_on_video(reader, args.input, args.output)
    else:
        run_on_image(reader, args.input, args.output)


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# Notes for Indian roads:
# - Indian plates: state(2) + RTO(1-2) + series(1-2) + number(1-4),
#   e.g. KA05EF9012, MH12AB3456, DL01CQ1234, TN10AA1234.
#   Private vehicles: black text on white. Commercial: black text on yellow.
# - Many plates carry an "IND" band + security hologram on the far left;
#   we crop it off before OCR.
# - OpenCV 5.x removed the legacy Haar CascadeClassifier, so detection is
#   contour-based (blackhat + Canny). No external model download needed.
# - For higher accuracy on angled/fast-moving plates, swap PlateReader.detect()
#   for a YOLOv8 model fine-tuned on Indian license plates; the rest of the
#   pipeline (crop -> strip IND -> preprocess -> OCR -> validate) stays the same.
# ---------------------------------------------------------------------------
