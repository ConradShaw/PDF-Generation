from reportlab.platypus import Flowable, Table, TableStyle, Paragraph
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

styles = getSampleStyleSheet()

class InfoPanel(Flowable):
    def __init__(self, content, style=None, background_color=colors.whitesmoke, padding=6, corner_radius=4):
        """
        A reusable panel for displaying text in PDFs.

        Args:
            content (str or Paragraph): The content to display.
            style (ParagraphStyle, optional): Style for the text. 
                If None, a default style is used.
        """
        super().__init__()

        # If the user passed a raw string and a style, wrap it in Paragraph
        default_style = getSampleStyleSheet()["Normal"]
        if isinstance(content, str):
            self.paragraph = Paragraph(content, style or default_style)
        elif isinstance(content, Paragraph):
            self.paragraph = content
        else:
            raise TypeError("InfoPanel content must be a string or Paragraph")

        """
        content: string or Paragraph
        style: ParagraphStyle for the text
        background_color: fill color behind text
        padding: space between text and rectangle border
        corner_radius: roundness of corners
        """
        
        self.style = style or default_style
        self.background_color = background_color
        self.padding = padding
        self.corner_radius = corner_radius    
    
    def wrap(self, availWidth, availHeight):
        # Paragraph size plus padding
        self.width, self.height = self.paragraph.wrap(availWidth - 2*self.padding, availHeight - 2*self.padding)
        self.width += 2*self.padding
        self.height += 2*self.padding
        return self.width, self.height

    def draw(self):
        # Draw the rounded rectangle first
        self.canv.setFillColor(self.background_color)
        self.canv.roundRect(0, 0, self.width, self.height, self.corner_radius, fill=1, stroke=0)

        # Draw the paragraph on top with padding
        self.paragraph.drawOn(self.canv, self.padding, self.padding)















