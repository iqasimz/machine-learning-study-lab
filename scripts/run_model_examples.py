"""Reproduce the bounded gallery locally, reusing but never overwriting completed runs."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from training_lab import model_lab as lab
from training_lab.artifacts import verify

EXAMPLES = [
 ("ml-digits-logistic-demo","ml","digits","classification","logistic",{}),
 ("ml-digits-forest-demo","ml","digits","classification","forest",{}),
 ("dl-digits-mlp-demo","dl","digits","classification","mlp",{"epochs":10}),
 ("dl-fashion-cnn-demo","dl","fashion_mnist","classification","cnn",{"epochs":5,"width":32,"batch_size":128}),
 ("dl-fashion-mlp-demo","dl","fashion_mnist","classification","mlp",{"epochs":5,"width":64,"batch_size":128}),
 ("ml-fashion-logistic-demo","ml","fashion_mnist","classification","logistic",{}),
 ("ml-red-wine-ridge-demo","ml","wine_red","regression","ridge",{}),
 ("ml-red-wine-boost-demo","ml","wine_red","regression","boost",{}),
 ("ml-clusters-kmeans-demo","ml","clusters","clustering","kmeans",{}),
 ("ml-digits-pca-demo","ml","digits","representation","pca",{}),
 ("dl-digits-autoencoder-demo","dl","digits","representation","autoencoder",{"epochs":10}),
]

if __name__=="__main__":
    for identifier,area,dataset,task,model,controls in EXAMPLES:
        directory=lab.RUNS/identifier
        if directory.exists():
            verify(directory)
            print(identifier,"already complete; retained",flush=True)
            continue
        manifest=lab.fit(identifier,dict(area=area,dataset=dataset,task=task,model=model,controls=controls))
        run=lab.detail(identifier)
        result=run.get("result",{})
        summary=result.get("predictions",{}).get("test",{}).get("scores")
        if summary is None:
            extra=result.get("representation",result.get("clustering",{}))
            summary={k:v for k,v in extra.items() if k not in {"projection","assignments","labels"}}
        print(identifier,manifest["status"],summary,flush=True)
