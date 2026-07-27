import numpy as np


class SketchAdapter:

    def __init__(self):
        self.debug = True

    def __call__(self,
                 sketch,
                 word_values=None,
                 word_phones=None):

        sketch = np.asarray(sketch, dtype=np.float32)

        if self.debug:
            self.report("Input Sketch", sketch)

        # ===========================
        # Stage 1
        # Boundary smoothing
        # ===========================
        sketch = self.boundary_smoothing(sketch)

        if self.debug:
            self.report("After Boundary Smoothing", sketch)

        return sketch.tolist()

    ####################################################################
    # Stage 1
    ####################################################################

    def boundary_smoothing(self, sketch):

        out = sketch.copy()

        if len(out) < 3:
            return out

        out[1:-1] = (
            0.25 * out[:-2] +
            0.50 * out[1:-1] +
            0.25 * out[2:]
        )

        return np.clip(out, 0.0, 1.0)

    ####################################################################
    # Debug
    ####################################################################

    def report(self, title, arr):

        # print("\n===================================================")
        # print(title)
        # print("===================================================")

        # print(np.round(arr, 3))

        # print("\nStatistics")

        # print(f"Length : {len(arr)}")
        # print(f"Min    : {arr.min():.3f}")
        # print(f"Max    : {arr.max():.3f}")
        # print(f"Mean   : {arr.mean():.3f}")
        # print(f"Std    : {arr.std():.3f}")

        if len(arr) > 1:
            diff = np.diff(arr)

            print(f"Largest Rise : {diff.max():.3f}")
            print(f"Largest Fall : {diff.min():.3f}")

        # print("===================================================\n")