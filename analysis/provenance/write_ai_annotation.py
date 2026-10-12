"""Blind pairwise annotation of human_check_sheet.csv by Claude (Anthropic AI model), done item by item by reading
each pair, WITHOUT access to human_check_KEY.csv. This is an AI annotation and must be reported as such --
it is NOT a human assessment.

better: which sentence is better English (more grammatical and natural); 'same' = tokenization-only difference
        (e.g. "I 'm" vs "I ' m", "have n't" vs "haven't") or equally flawed.
meaning_same: whether the two sentences express the same literal meaning."""
import csv

J = {  # item: (better, meaning_same)
0: ("1", "yes"), 1: ("2", "yes"), 2: ("same", "yes"), 3: ("2", "yes"), 4: ("2", "yes"), 5: ("same", "yes"),
6: ("2", "yes"), 7: ("1", "yes"), 8: ("1", "yes"), 9: ("1", "yes"), 10: ("2", "yes"), 11: ("same", "yes"),
12: ("same", "yes"), 13: ("1", "yes"), 14: ("same", "yes"), 15: ("2", "no"), 16: ("2", "yes"), 17: ("1", "yes"),
18: ("2", "yes"), 19: ("same", "yes"), 20: ("2", "yes"), 21: ("1", "yes"), 22: ("2", "yes"), 23: ("same", "yes"),
24: ("2", "yes"), 25: ("2", "yes"), 26: ("2", "yes"), 27: ("same", "yes"), 28: ("2", "yes"), 29: ("2", "yes"),
30: ("2", "yes"), 31: ("1", "yes"), 32: ("2", "no"), 33: ("1", "yes"), 34: ("1", "no"), 35: ("2", "yes"),
36: ("2", "yes"), 37: ("1", "yes"), 38: ("1", "yes"), 39: ("1", "yes"), 40: ("2", "yes"), 41: ("2", "yes"),
42: ("2", "no"), 43: ("2", "yes"), 44: ("1", "yes"), 45: ("same", "yes"), 46: ("1", "yes"), 47: ("2", "yes"),
48: ("same", "yes"), 49: ("same", "yes"), 50: ("2", "yes"), 51: ("2", "yes"), 52: ("2", "yes"), 53: ("2", "yes"),
54: ("1", "yes"), 55: ("2", "yes"), 56: ("1", "yes"), 57: ("2", "yes"), 58: ("2", "yes"), 59: ("1", "yes"),
60: ("1", "yes"), 61: ("2", "yes"), 62: ("1", "yes"), 63: ("2", "yes"), 64: ("1", "yes"), 65: ("1", "yes"),
66: ("1", "yes"), 67: ("1", "yes"), 68: ("2", "yes"), 69: ("1", "yes"), 70: ("2", "yes"), 71: ("2", "yes"),
72: ("1", "yes"), 73: ("1", "yes"), 74: ("2", "no"), 75: ("1", "yes"), 76: ("2", "yes"), 77: ("2", "yes"),
78: ("2", "yes"), 79: ("2", "yes"), 80: ("1", "yes"), 81: ("2", "yes"), 82: ("1", "yes"), 83: ("1", "yes"),
84: ("1", "yes"), 85: ("2", "yes"), 86: ("1", "yes"), 87: ("1", "yes"), 88: ("2", "yes"), 89: ("2", "yes"),
90: ("1", "yes"), 91: ("2", "yes"), 92: ("1", "yes"), 93: ("2", "yes"), 94: ("2", "yes"), 95: ("2", "yes"),
96: ("2", "yes"), 97: ("same", "yes"), 98: ("1", "yes"), 99: ("1", "yes"), 100: ("1", "yes"), 101: ("1", "yes"),
102: ("2", "yes"), 103: ("1", "yes"), 104: ("1", "yes"), 105: ("1", "yes"), 106: ("1", "yes"), 107: ("same", "yes"),
108: ("1", "yes"), 109: ("same", "yes"), 110: ("1", "yes"), 111: ("2", "yes"), 112: ("1", "yes"), 113: ("1", "yes"),
114: ("same", "no"), 115: ("1", "yes"), 116: ("2", "yes"), 117: ("2", "yes"), 118: ("1", "yes"), 119: ("2", "yes"),
120: ("2", "yes"), 121: ("1", "yes"), 122: ("same", "yes"), 123: ("1", "yes"), 124: ("1", "yes"), 125: ("2", "yes"),
126: ("1", "yes"), 127: ("1", "yes"), 128: ("same", "no"), 129: ("2", "yes"), 130: ("2", "yes"), 131: ("2", "yes"),
132: ("1", "yes"), 133: ("2", "yes"), 134: ("same", "yes"), 135: ("1", "yes"), 136: ("2", "yes"), 137: ("2", "yes"),
138: ("2", "yes"), 139: ("1", "yes"), 140: ("2", "yes"), 141: ("2", "yes"), 142: ("1", "yes"), 143: ("2", "yes"),
144: ("2", "yes"), 145: ("1", "yes"), 146: ("2", "yes"), 147: ("1", "yes"), 148: ("2", "yes"), 149: ("1", "yes"),
}
assert sorted(J) == list(range(150))
with open("ai_annotation_claude.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["item", "better (1/2/same)", "meaning_same (yes/no)", "annotator"])
    for i in range(150):
        w.writerow([i, J[i][0], J[i][1], "Claude (Anthropic AI model), blind to the key -- AI annotation, not human"])
print("wrote ai_annotation_claude.csv (150 items)")
