@echo off
echo Compiling LaTeX paper...
echo.

echo Step 1: First compilation
pdflatex chest_xray_resolution_paper.tex
if %errorlevel% neq 0 (
    echo Error in first compilation
    pause
    exit /b 1
)

echo.
echo Step 2: Processing bibliography
bibtex chest_xray_resolution_paper
if %errorlevel% neq 0 (
    echo Error in bibliography processing
    pause
    exit /b 1
)

echo.
echo Step 3: Second compilation
pdflatex chest_xray_resolution_paper.tex
if %errorlevel% neq 0 (
    echo Error in second compilation
    pause
    exit /b 1
)

echo.
echo Step 4: Final compilation
pdflatex chest_xray_resolution_paper.tex
if %errorlevel% neq 0 (
    echo Error in final compilation
    pause
    exit /b 1
)

echo.
echo Compilation completed successfully!
echo Output: chest_xray_resolution_paper.pdf
echo.
pause
