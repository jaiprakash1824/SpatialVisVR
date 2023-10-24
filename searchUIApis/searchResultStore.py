import tifffile as tiff
import os
class SearchResultStore:
    _instance = None
    _cache = {}
    _cache_channel_names = {}
    _image_Path = []
    _channel_Names = []
    _basePath = "/Volumes/Jai_Lab_T7/Data/Codex/CRC_TMA_A_hyperstacks_bestFocus/bestFocus"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SearchResultStore, cls).__new__(cls)
        return cls._instance
    
    # path needs to be list of file paths list 
    def set_imagePath(self, path):
        res = []
        for i in path:
            k = os.path.join(self._basePath,i)
            res.append(k)
        self._image_Path = res
    
    def get_imagePath(self):
        return self._image_Path
        
    def get_channel_names(self):
        return ["HOECHST1","blank","blank","blank","HOECHST2","CD44","FOXP3","CDX2","HOECHST3","CD8","p53","GATA3","HOECHST4","CD45","T-bet","beta-catenin","HOECHST5","HLA-DR","PD-L1","Ki67","HOECHST6","CD45RA","CD4","CD21","HOECHST7","MUC-1","CD30","CD2","HOECHST8","Vimentin","CD20","LAG-3","HOECHST9","Na-K-ATPase","CD5","IDO-1","HOECHST10","Cytokeratin","CD11b","CD56","HOECHST11","aSMA","BCL-2","CD25","HOECHST12","Collagen IV","CD11c","PD-1","HOCHST13","Granzyme B","EGFR","VISTA","HOECHST14","CD15","CD194","ICOS","HOECHST15","MMP9","Synaptophysin","CD71","HOECHST16","GFAP","CD7","CD3","HOECHST17","Chromogranin A","CD163","CD57","HOECHST18","empty","CD45RO","CD68","HOECHST19","empty","CD31","Podoplanin","HOECHST20","empty","CD34","CD38","HOECHST21","empty","CD138","MMP12","HOECHST22","empty","empty","empty","HOECHST23","empty","empty","DRAQ5"]

    def get_cached_images(self):
        return list(self._cache.keys())
    
    def get_cached_image_data(self, path):
        return self._cache[path]

    def clear_cache(self):
        self._cache.clear()

    def _get_channel_names(self,_):
        return ["HOECHST1","blank","blank","blank","HOECHST2","CD44","FOXP3","CDX2","HOECHST3","CD8","p53","GATA3","HOECHST4","CD45","T-bet","beta-catenin","HOECHST5","HLA-DR","PD-L1","Ki67","HOECHST6","CD45RA","CD4","CD21","HOECHST7","MUC-1","CD30","CD2","HOECHST8","Vimentin","CD20","LAG-3","HOECHST9","Na-K-ATPase","CD5","IDO-1","HOECHST10","Cytokeratin","CD11b","CD56","HOECHST11","aSMA","BCL-2","CD25","HOECHST12","Collagen IV","CD11c","PD-1","HOCHST13","Granzyme B","EGFR","VISTA","HOECHST14","CD15","CD194","ICOS","HOECHST15","MMP9","Synaptophysin","CD71","HOECHST16","GFAP","CD7","CD3","HOECHST17","Chromogranin A","CD163","CD57","HOECHST18","empty","CD45RO","CD68","HOECHST19","empty","CD31","Podoplanin","HOECHST20","empty","CD34","CD38","HOECHST21","empty","CD138","MMP12","HOECHST22","empty","empty","empty","HOECHST23","empty","empty","DRAQ5"]

