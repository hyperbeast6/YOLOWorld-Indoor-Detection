# YOLO-World S Model Missing Files Download Script
# For Windows PowerShell

Write-Host "Starting to download missing files for YOLO-World S model..." -ForegroundColor Green
Write-Host ""

# Check if huggingface-cli is installed
try {
    $null = Get-Command huggingface-cli -ErrorAction Stop
    Write-Host "Detected huggingface-cli is installed" -ForegroundColor Green
} catch {
    Write-Host "huggingface-cli not detected, please install first:" -ForegroundColor Red
    Write-Host "   pip install huggingface-hub" -ForegroundColor Yellow
    Write-Host "   or: conda install -c conda-forge huggingface-hub" -ForegroundColor Yellow
    exit 1
}

# Set mirror
$env:HF_ENDPOINT = "https://hf-mirror.com"
Write-Host "Set Hugging Face mirror: $env:HF_ENDPOINT" -ForegroundColor Green
Write-Host ""

# Download CLIP model
Write-Host "Downloading CLIP text encoder model..." -ForegroundColor Yellow
Write-Host "   Model: openai/clip-vit-base-patch32" -ForegroundColor Gray
Write-Host "   Size: ~150-200MB" -ForegroundColor Gray
Write-Host "   Location: ./weights/clip-vit-base-patch32/" -ForegroundColor Gray
Write-Host ""

try {
    huggingface-cli download --resume-download openai/clip-vit-base-patch32 --local-dir ./weights/clip-vit-base-patch32 --local-dir-use-symlinks False
    if (Test-Path "./weights/clip-vit-base-patch32") {
        Write-Host "CLIP model download completed" -ForegroundColor Green
    } else {
        throw "File not found after download"
    }
} catch {
    Write-Host "CLIP model download failed: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Create necessary directories
Write-Host "Creating necessary directories..." -ForegroundColor Yellow
try {
    New-Item -ItemType Directory -Force -Path "./data/coco/lvis" | Out-Null
    New-Item -ItemType Directory -Force -Path "./data/texts" | Out-Null
    Write-Host "Directory creation completed" -ForegroundColor Green
} catch {
    Write-Host "Directory creation failed: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Download LVIS validation annotation file
Write-Host "Downloading LVIS validation annotation file..." -ForegroundColor Yellow
Write-Host "   File: lvis_v1_minival_inserted_image_name.json" -ForegroundColor Gray
Write-Host "   Size: few MB" -ForegroundColor Gray
Write-Host "   Location: ./data/coco/lvis/" -ForegroundColor Gray
Write-Host ""

try {
    huggingface-cli download GLIPModel/GLIP lvis_v1_minival_inserted_image_name.json --local-dir ./data/coco/lvis --local-dir-use-symlinks False --resume-download
    if (Test-Path "E:\code\YOLO-World\data\coco\lvis\lvis_v1_minival_inserted_image_name.json") {
        Write-Host "LVIS annotation file download completed" -ForegroundColor Green
    } else {
        throw "File not found after download"
    }
} catch {
    Write-Host "LVIS annotation file download failed: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Download Objects365 class text file
Write-Host "Downloading Objects365 class text file..." -ForegroundColor Yellow
Write-Host "   File: obj365v1_class_texts.json" -ForegroundColor Gray
Write-Host "   Size: few MB" -ForegroundColor Gray
Write-Host "   Location: ./data/texts/" -ForegroundColor Gray
Write-Host ""

try {
    huggingface-cli download wondervictor/YOLO-World data/texts/obj365v1_class_texts.json --local-dir ./data/texts --local-dir-use-symlinks False --resume-download
    if (Test-Path "./data/texts/obj365v1_class_texts.json") {
        Write-Host "Objects365 class text download completed" -ForegroundColor Green
    } else {
        throw "File not found after download"
    }
} catch {
    Write-Host "Objects365 class text download failed: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Download LVIS class text file
Write-Host "Downloading LVIS class text file..." -ForegroundColor Yellow
Write-Host "   File: lvis_v1_class_texts.json" -ForegroundColor Gray
Write-Host "   Size: few MB" -ForegroundColor Gray
Write-Host "   Location: ./data/texts/" -ForegroundColor Gray
Write-Host ""

try {
    huggingface-cli download wondervictor/YOLO-World data/texts/lvis_v1_class_texts.json --local-dir ./data/texts --local-dir-use-symlinks False --resume-download
    if (Test-Path "./data/texts/lvis_v1_class_texts.json") {
        Write-Host "LVIS class text download completed" -ForegroundColor Green
    } else {
        throw "File not found after download"
    }
} catch {
    Write-Host "LVIS class text download failed: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "All files download completed!" -ForegroundColor Green
Write-Host ""

# Display downloaded file list
Write-Host "Downloaded files list:" -ForegroundColor Cyan
Write-Host ""

$files = @(
    @{Path="./weights/clip-vit-base-patch32/"; Description="CLIP Text Encoder Model"}
    @{Path="E:\code\YOLO-World\data\coco\lvis\lvis_v1_minival_inserted_image_name.json"; Description="LVIS Validation Annotation File"}
    @{Path="./data/texts/obj365v1_class_texts.json"; Description="Objects365 Class Text File"}
    @{Path="./data/texts/lvis_v1_class_texts.json"; Description="LVIS Class Text File"}
)

foreach ($file in $files) {
    if (Test-Path $file.Path) {
        Write-Host "  [OK] $($file.Description)" -ForegroundColor Green
    } else {
        Write-Host "  [FAIL] $($file.Description)" -ForegroundColor Red
    }
    Write-Host "      Path: $($file.Path)" -ForegroundColor Gray
}

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "   1. Ensure your YOLO-World S model weight file is in weights/ directory" -ForegroundColor White
Write-Host "   2. Run: python demo/simple_demo.py" -ForegroundColor White
Write-Host ""

Write-Host "Tip: If download is interrupted, you can re-run this script, it will auto-resume" -ForegroundColor Cyan
