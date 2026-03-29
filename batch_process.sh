#!/bin/bash

##############################################################################
#  Batch Process Images Through GAN Pipeline
#
#  Workflow:
#  1. For each image in data/ground_truth/
#  2. Run inference_pipeline.py
#  3. Find the generated output file
#  4. Rename to match ground truth filename
#  5. Generate comparison grid
#
#  Usage:
#      bash batch_process.sh [--limit N] [--skip-metrics]
#
#  Examples:
#      bash batch_process.sh              # process all
#      bash batch_process.sh --limit 5    # process first 5
#      bash batch_process.sh --skip-metrics  # skip metrics computation
##############################################################################

set -e  # exit on error

# ────────────────── CONFIG ──────────────────
GT_DIR="data/ground_truth"
OUTPUT_DIR="outputs/generated"
VISUALS_DIR="outputs/visuals"
SKIP_METRICS=false
LIMIT=0
PROCESSED=0

# ────────────────── PARSE ARGS ──────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --skip-metrics)
            SKIP_METRICS=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ────────────────── SETUP ──────────────────
mkdir -p "$OUTPUT_DIR" "$VISUALS_DIR"
echo "=========================================="
echo "Batch Processing GAN Pipeline"
echo "=========================================="
echo "Ground truth: $GT_DIR"
echo "Output dir:   $OUTPUT_DIR"
echo "Visuals:      $VISUALS_DIR"
echo ""

# Check if ground_truth dir exists
if [ ! -d "$GT_DIR" ]; then
    echo "❌ Error: $GT_DIR does not exist"
    exit 1
fi

# Count images
TOTAL=$(find "$GT_DIR" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) | wc -l)
if [ "$TOTAL" -eq 0 ]; then
    echo "❌ Error: No images found in $GT_DIR"
    exit 1
fi

if [ "$LIMIT" -gt 0 ] && [ "$LIMIT" -lt "$TOTAL" ]; then
    TOTAL=$LIMIT
    echo "Processing first $TOTAL images (--limit $LIMIT)"
else
    echo "Processing $TOTAL images"
fi
echo ""

# ────────────────── MAIN LOOP ──────────────────
for gt_img in $(find "$GT_DIR" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) | sort | head -$TOTAL); do
    fname=$(basename "$gt_img")
    fname_noext="${fname%.*}"
    
    echo "────────────────────────────────────────"
    echo "Processing: $fname [$((PROCESSED+1))/$TOTAL]"
    echo "────────────────────────────────────────"
    
    # 1. Run pipeline on full GT image (will be downscaled)
    echo "[Pipeline] Running inference..."
    python3 inference_pipeline.py "$gt_img" --output-dir "$OUTPUT_DIR/" 2>&1 | head -20
    
    # 2. Find latest generated file (pipeline creates colorized_*.jpg)
    echo "[Pipeline] Finding generated output..."
    latest=$(ls -t "$OUTPUT_DIR"/colorized_* 2>/dev/null | head -n 1)
    
    if [ -z "$latest" ]; then
        echo "❌ Error: No output file generated for $fname"
        continue
    fi
    
    echo "   Generated: $(basename $latest)"
    
    # 3. Rename to match ground truth filename
    target_path="$OUTPUT_DIR/$fname"
    echo "[Pipeline] Renaming → $fname"
    mv "$latest" "$target_path"
    echo "   ✓ Saved: $target_path"
    
    # 4. Generate comparison grid
    echo "[Visualize] Creating comparison grid..."
    python3 visualize.py --image "$fname" 2>&1 | grep -E "Processing:|Saved:|Error:" || true
    
    PROCESSED=$((PROCESSED+1))
    echo ""
done

# ────────────────── SUMMARY ──────────────────
echo "=========================================="
echo "BATCH PROCESSING COMPLETE"
echo "=========================================="
echo "✓ Processed: $PROCESSED/$TOTAL images"
echo "✓ Output:    $OUTPUT_DIR/"
echo "✓ Grids:     $VISUALS_DIR/"
echo ""

# ────────────────── METRICS ──────────────────
if [ "$SKIP_METRICS" = false ]; then
    echo "Computing metrics..."
    python3 metrics.py
else
    echo "[Metrics] Skipped (use --skip-metrics to disable)"
fi

echo ""
echo "📊 Next Steps:"
echo "   1. Review grids in: outputs/visuals/"
echo "   2. Check metrics output above"
echo "   3. Analyze generated images in: outputs/generated/"
echo "=========================================="
