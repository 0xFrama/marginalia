from docling.document_converter import DocumentConverter


converter = DocumentConverter()
doc = converter.convert("samples/attention.pdf").document

for item, level in doc.iterate_items():
    if item.label == "table":
        print(dir(item))
        print(item.export_to_markdown())
        break
