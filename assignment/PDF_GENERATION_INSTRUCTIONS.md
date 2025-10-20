# Instructions to Generate PDF

The assignment brief PDF needs to be generated from `DKR_HW_Combined_F25.md`.

## Option 1: Using Pandoc (Recommended)

```bash
# Install pandoc if not already installed
brew install pandoc basictex

# Generate PDF
cd assignment
pandoc DKR_HW_Combined_F25.md -o DKR_HW_Combined_F25.pdf \
  --pdf-engine=xelatex \
  --variable geometry:margin=1in \
  --variable fontsize=11pt \
  --toc \
  --number-sections

# Commit
git add DKR_HW_Combined_F25.pdf
git commit -m "Add PDF version of assignment brief"
git push origin main
```

## Option 2: Using Online Tool

1. Go to https://www.markdowntopdf.com/ or https://md2pdf.netlify.app/
2. Upload `DKR_HW_Combined_F25.md`
3. Download the generated PDF as `DKR_HW_Combined_F25.pdf`
4. Save to `assignment/` directory
5. Commit and push

## Option 3: Using VS Code Extension

1. Install "Markdown PDF" extension in VS Code
2. Open `DKR_HW_Combined_F25.md`
3. Right-click → "Markdown PDF: Export (pdf)"
4. Rename to `DKR_HW_Combined_F25.pdf`
5. Commit and push

**After generating, delete this file!**
