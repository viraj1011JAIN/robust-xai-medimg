Param(
  [string]$Config = "configs/tiny.yaml",
  [string]$Out    = "docs/figures/gradcam_tiny.png"
)
$ArgsList = @("-m","src.xai.export","--config",$Config,"--out",$Out)
python @ArgsList
