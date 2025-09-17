## ⚡ Default Preparation

To help you get started quickly, we’ve included a set of **pre-trained networks** ready for use.  
These models allow you to **predict new views** and even **generate videos** right away — no training required.  

👉 **Note**: These networks are trained exclusively on the **Lego dataset**.  

If you’d like to **continue training** or fine-tune the models, you’ll first need to generate the Lego data using the CLI:  

```bash
python main.py train --data lego.h5
```

🧠 Pre-Trained Networks Available
* `coarse_finki_nerf.weights`
* `fine_finki_nerf.weights`

Use them as a starting point, or build upon them for your own experiments.

> NOTE : Everything that will be generated and used by the CLI remains here.
