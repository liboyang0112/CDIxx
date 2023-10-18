pdflatex "\def\formula{\boldsymbol{$1}}\input{$CDI_DIR/config/formulascript.tex}" > /dev/null
mv formulascript.pdf out_$2.pdf
rm formulascript.*
pdftoppm out_$2.pdf -png > out_$2.png
