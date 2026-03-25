# Net2Vec + TCAV: Spatial Encoding vs Causal Influence

Reimplements **Net2Vec** (Fong & Vedaldi, CVPR 2018) on AlexNet and
introduces a novel comparison with **TCAV** (Kim et al., ICML 2018).

## Research question

Net2Vec measures how well a filter spatially encodes a concept (IoU).
TCAV measures how causally influential a concept is for a prediction.
Do they agree? This project finds they are largely **dissociated**
(r = -0.02), suggesting the two methods capture complementary aspects
of concept representation.

## Reproducing the results

### On Google Colab (recommended)

Open a new Colab notebook and run:
```python
!git clone https://github.com/YOUR_USERNAME/net2vec-tcav.git
!bash run_colab.sh
```

This downloads Broden (~930 MB), builds the concept index, runs both
experiments, and generates all figures. Total runtime ~2 hours on a
T4 GPU.

### Locally
```bash
git clone https://github.com/YOUR_USERNAME/net2vec-tcav.git
cd net2vec-tcav
pip install -r requirements.txt
# Download Broden manually to ../broden1_227
bash run_colab.sh
```

## Repo structure
```
src/
    models.py 
    dataset.py  
    net2vec.py     
    tcav.py         
experiments/
    build_index.py  
    run_baseline.py 
    run_tcav.py 
    plot_results.py 
results/
    net2vec_results.pkl  
    tcav_results.pkl     
figures/
    fig1–fig4         
```

## Key results

| Concept  | Net2Vec IoU | TCAV score |
|----------|------------|------------|
| grass    | 0.047      | 0.898      |
| water    | 0.124      | 0.148      |
| wheel    | 0.000      | 0.757      |
| sky      | 0.012      | 0.153      |

Pearson correlation between IoU and TCAV: **r = -0.02, p = 0.96**

