#!/bin/bash
# Quick diagnostic script to check disk space configuration

echo "=========================================="
echo "Disk Space Diagnostic"
echo "=========================================="
echo ""

echo "1. TMPDIR environment variable:"
if [ -z "$TMPDIR" ]; then
    echo "   TMPDIR is NOT set (using default: $(python3 -c 'import tempfile; print(tempfile.gettempdir())'))"
    TMPDIR=$(python3 -c 'import tempfile; print(tempfile.gettempdir())')
else
    echo "   TMPDIR=$TMPDIR"
fi
echo ""

echo "2. Disk space in temp directory ($TMPDIR):"
df -h "$TMPDIR" | tail -n 1 | awk '{print "   Total: "$2", Used: "$3", Available: "$4", Use%: "$5}'
echo ""

echo "3. All filesystem usage:"
df -h | grep -v "tmpfs\|loop"
echo ""

echo "4. Largest directories in temp:"
du -sh "$TMPDIR"/* 2>/dev/null | sort -h | tail -n 10
echo ""

echo "5. TensorFlow temp file locations (if any exist):"
find "$TMPDIR" -name "tmp*" -type f 2>/dev/null | head -n 5
echo ""

echo "6. Recommended fix:"
echo "   If available space is < 10GB, try:"
echo "   export TMPDIR=/path/to/disk/with/space"
echo "   mkdir -p \$TMPDIR"
echo ""
