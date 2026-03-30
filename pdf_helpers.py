from reportlab.platypus import Flowable, Table, TableStyle, Paragraph
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

styles = getSampleStyleSheet()

class InfoPanel(Flowable):
    def __init__(self, content, style=None):
        """
        A reusable panel for displaying text in PDFs.

        Args:
            content (str or Paragraph): The content to display.
            style (ParagraphStyle, optional): Style for the text. 
                If None, a default style is used.
        """
        super().__init__()

        # If the user passed a raw string and a style, wrap it in Paragraph
        if isinstance(content, str):
            from reportlab.lib.styles import getSampleStyleSheet
            default_style = getSampleStyleSheet()["Normal"]
            self.paragraph = Paragraph(content, style or default_style)
        elif isinstance(content, Paragraph):
            self.paragraph = content
        else:
            raise TypeError("InfoPanel content must be a string or Paragraph")

        # Save style (optional, can be used for layout calculations)
        self.style = style or default_style

    def wrap(self, availWidth, availHeight):
        """Calculate required width and height"""
        return self.paragraph.wrap(availWidth, availHeight)

    def draw(self):
        """Draw the paragraph on the canvas"""
        self.paragraph.drawOn(self.canv, 0, 0)
