#!/bin/bash

echo "Compiling LaTeX paper..."
echo

echo "Step 1: First compilation"
pdflatex chest_xray_resolution_paper.tex
if [ $? -ne 0 ]; then
    echo "Error in first compilation"
    exit 1
fi

echo
echo "Step 2: Processing bibliography"
bibtex chest_xray_resolution_paper
if [ $? -ne 0 ]; then
    echo "Error in bibliography processing"
    exit 1
fi

echo
echo "Step 3: Second compilation"
pdflatex chest_xray_resolution_paper.tex
if [ $? -ne 0 ]; then
    echo "Error in second compilation"
    exit 1
fi

echo
echo "Step 4: Final compilation"
pdflatex chest_xray_resolution_paper.tex
if [ $? -ne 0 ]; then
    echo "Error in final compilation"
    exit 1
fi

echo
echo "Compilation completed successfully!"
echo "Output: chest_xray_resolution_paper.pdf"
echo
