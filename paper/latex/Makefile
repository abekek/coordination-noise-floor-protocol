LATEX = latexmk
LATEXFLAGS = -pdf -interaction=nonstopmode -file-line-error

.PHONY: all clean watch

all: et-mcp.pdf

et-mcp.pdf: et-mcp.tex $(wildcard sections/*.tex) references.bib
	$(LATEX) $(LATEXFLAGS) et-mcp.tex

watch:
	$(LATEX) $(LATEXFLAGS) -pvc et-mcp.tex

clean:
	$(LATEX) -C
	rm -f *.aux *.bbl *.blg *.log *.out *.fdb_latexmk *.fls *.synctex.gz \
	      *.toc missfont.log
