from reportlab.platypus import Flowable, Table, TableStyle, Paragraph
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

styles = getSampleStyleSheet()

class InfoPanel(Flowable):
    """
    Simple info panel for ReportLab PDF.
    Accepts either a string or a list-of-lists of Paragraphs/Tables.
    """
    def __init__(self, content):
        super().__init__()
        # Wrap string content in a Paragraph inside a single-cell table
        if isinstance(content, str):
            self.content = [[Paragraph(content, styles['Normal'])]]
        else:
            self.content = content  # assume already a Table-compatible list-of-lists

    def wrap(self, availWidth, availHeight):
        # Set width and default height
        self.width = availWidth
        self.height = 50  # default panel height; you can adjust as needed
        return (self.width, self.height)

    def draw(self):
        table = Table(self.content, colWidths=self.width)
        table.setStyle(TableStyle([
            ('BOX', (0,0), (-1,-1), 1, colors.black),
            ('BACKGROUND', (0,0), (-1,-1), colors.whitesmoke),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING', (0,0), (-1,-1), 6),
            ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ]))
        table.wrapOn(self.canv, self.width, self.height)
        table.drawOn(self.canv, 0, 0)
