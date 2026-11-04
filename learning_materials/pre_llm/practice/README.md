# Small calculations you can inspect

These files use only Python's standard library. They do not download a model, call an API or train anything. All inputs are teaching examples, not experimental measurements.

From this directory:

```bash
python3 worked_math.py
python3 -m unittest -v test_worked_math.py
```

If Node.js is already installed, you can additionally test the visual lab's maths and initial event wiring:

```bash
node test_visual_lab.cjs
```

No Node packages are required. This is not a replacement for visually opening the HTML in a browser.

## Exercises before changing the code

1. Calculate the printed softmax, loss and attention values by hand.
2. Change the target class, not the logits. Explain why probabilities stay fixed while loss changes.
3. Give all GRPO responses the same reward. Explain the zero centred signal.
4. Double context/concurrency in the KV example. Predict bytes first.
5. Add one test for a malformed input. Explain the intended exception.

The bootstrap demonstration is tiny and pedagogical. It is not an inferential result about any real experiment. Functions validate their intended toy inputs; they are not production numerical libraries.
