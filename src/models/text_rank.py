"""Explicitly tunable PCA rank; rank32 preserves the original transformation."""
from sklearn.decomposition import PCA
from src.models.text_preprocessing import TextPreprocessor


class TextRankPreprocessor(TextPreprocessor):
    def __init__(self,rank=32):
        self.rank=rank

    def fit(self,X,y=None):
        if self.rank not in [8,16,32,64]:raise ValueError('Unregistered PCA rank')
        super().fit(X,y)
        if self.other_ or not self.text_:raise ValueError('Text-only inputs required')
        if self.rank!=32:
            self.pca_=PCA(n_components=min(self.rank,len(X)-1,len(self.text_)),svd_solver='randomized',
                          random_state=42,whiten=True).fit(self.scale_.transform(X[self.text_]))
            self.columns_=[f'text__pc{i}' for i in range(self.pca_.n_components_)]
        return self
