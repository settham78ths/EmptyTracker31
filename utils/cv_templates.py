
import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.graphics.shapes import Drawing, Rect, Line
from reportlab.platypus.flowables import Flowable
import base64

class ColorBox(Flowable):
    """Custom flowable for colored boxes"""
    def __init__(self, width, height, color):
        self.width = width
        self.height = height
        self.color = color

    def draw(self):
        self.canv.setFillColor(self.color)
        self.canv.rect(0, 0, self.width, self.height, fill=1)

class CVTemplateGenerator:
    """Generate professional CV templates with different designs"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
    
    def setup_custom_styles(self):
        """Setup custom paragraph styles for different templates"""
        
        # Modern Blue Template Styles
        self.modern_title = ParagraphStyle(
            'ModernTitle',
            parent=self.styles['Heading1'],
            fontSize=28,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=10,
            alignment=1,
            fontName='Helvetica-Bold'
        )
        
        self.modern_subtitle = ParagraphStyle(
            'ModernSubtitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#3498db'),
            spaceAfter=20,
            alignment=1,
            fontName='Helvetica'
        )
        
        self.section_header = ParagraphStyle(
            'SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=12,
            spaceBefore=20,
            fontName='Helvetica-Bold',
            borderWidth=0,
            borderColor=colors.HexColor('#3498db'),
            borderPadding=5
        )
        
        # Creative Template Styles
        self.creative_title = ParagraphStyle(
            'CreativeTitle',
            parent=self.styles['Heading1'],
            fontSize=26,
            textColor=colors.HexColor('#e74c3c'),
            spaceAfter=8,
            alignment=0,
            fontName='Helvetica-Bold'
        )
        
        # Executive Template Styles
        self.executive_title = ParagraphStyle(
            'ExecutiveTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#34495e'),
            spaceAfter=12,
            alignment=1,
            fontName='Times-Bold'
        )
        
        # Minimalist Template Styles
        self.minimal_title = ParagraphStyle(
            'MinimalTitle',
            parent=self.styles['Heading1'],
            fontSize=22,
            textColor=colors.black,
            spaceAfter=15,
            alignment=0,
            fontName='Helvetica-Light'
        )
    
    def generate_modern_blue_cv(self, cv_data):
        """Generate modern blue professional CV template"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, 
                               topMargin=2*cm, bottomMargin=2*cm)
        story = []
        
        # Header with blue accent
        story.append(ColorBox(doc.width, 0.5*cm, colors.HexColor('#3498db')))
        story.append(Spacer(1, 0.3*cm))
        
        # Name and title
        name = f"{cv_data.get('firstName', '')} {cv_data.get('lastName', '')}".strip()
        story.append(Paragraph(name, self.modern_title))
        
        job_title = cv_data.get('jobTitle', '')
        if job_title:
            story.append(Paragraph(job_title, self.modern_subtitle))
        
        # Contact info in table
        contact_data = []
        contact_info = []
        if cv_data.get('email'):
            contact_info.append(f"✉ {cv_data['email']}")
        if cv_data.get('phone'):
            contact_info.append(f"📞 {cv_data['phone']}")
        if cv_data.get('city'):
            contact_info.append(f"📍 {cv_data['city']}")
        if cv_data.get('linkedin'):
            contact_info.append(f"🔗 {cv_data['linkedin']}")
        
        if contact_info:
            # Split into two columns
            half = len(contact_info) // 2
            left_col = contact_info[:half]
            right_col = contact_info[half:]
            
            max_rows = max(len(left_col), len(right_col))
            for i in range(max_rows):
                left = left_col[i] if i < len(left_col) else ""
                right = right_col[i] if i < len(right_col) else ""
                contact_data.append([left, right])
            
            contact_table = Table(contact_data, colWidths=[doc.width/2, doc.width/2])
            contact_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#7f8c8d')),
            ]))
            story.append(contact_table)
        
        story.append(Spacer(1, 0.5*cm))
        
        # Professional summary with blue accent
        if cv_data.get('summary'):
            story.append(ColorBox(doc.width, 0.2*cm, colors.HexColor('#ecf0f1')))
            story.append(Spacer(1, 0.2*cm))
            story.append(Paragraph("PROFIL ZAWODOWY", self.section_header))
            summary_style = ParagraphStyle(
                'Summary',
                parent=self.styles['Normal'],
                fontSize=11,
                textColor=colors.HexColor('#2c3e50'),
                alignment=4,  # Justify
                spaceAfter=15
            )
            story.append(Paragraph(cv_data['summary'], summary_style))
        
        # Experience section
        experiences = cv_data.get('experiences', [])
        if any(exp.get('title') or exp.get('company') for exp in experiences):
            story.append(Paragraph("DOŚWIADCZENIE ZAWODOWE", self.section_header))
            
            for exp in experiences:
                if exp.get('title') or exp.get('company'):
                    # Experience header
                    exp_title = exp.get('title', 'Stanowisko')
                    exp_company = exp.get('company', 'Firma')
                    
                    title_style = ParagraphStyle(
                        'ExpTitle',
                        parent=self.styles['Normal'],
                        fontSize=12,
                        textColor=colors.HexColor('#2c3e50'),
                        fontName='Helvetica-Bold',
                        spaceAfter=3
                    )
                    
                    company_style = ParagraphStyle(
                        'ExpCompany',
                        parent=self.styles['Normal'],
                        fontSize=11,
                        textColor=colors.HexColor('#3498db'),
                        fontName='Helvetica-Bold',
                        spaceAfter=5
                    )
                    
                    story.append(Paragraph(exp_title, title_style))
                    story.append(Paragraph(exp_company, company_style))
                    
                    # Dates
                    start_date = exp.get('startDate', '')
                    end_date = exp.get('endDate', 'obecnie')
                    if start_date:
                        date_style = ParagraphStyle(
                            'ExpDate',
                            parent=self.styles['Normal'],
                            fontSize=10,
                            textColor=colors.HexColor('#7f8c8d'),
                            spaceAfter=8
                        )
                        story.append(Paragraph(f"{start_date} - {end_date}", date_style))
                    
                    # Description
                    if exp.get('description'):
                        desc_style = ParagraphStyle(
                            'ExpDesc',
                            parent=self.styles['Normal'],
                            fontSize=10,
                            textColor=colors.HexColor('#2c3e50'),
                            leftIndent=20,
                            spaceAfter=15
                        )
                        story.append(Paragraph(f"• {exp['description']}", desc_style))
        
        # Education section
        education = cv_data.get('education', [])
        if any(edu.get('degree') or edu.get('school') for edu in education):
            story.append(Paragraph("WYKSZTAŁCENIE", self.section_header))
            
            for edu in education:
                if edu.get('degree') or edu.get('school'):
                    degree = edu.get('degree', 'Kierunek')
                    school = edu.get('school', 'Uczelnia')
                    
                    edu_style = ParagraphStyle(
                        'Education',
                        parent=self.styles['Normal'],
                        fontSize=11,
                        textColor=colors.HexColor('#2c3e50'),
                        spaceAfter=8
                    )
                    
                    story.append(Paragraph(f"<b>{degree}</b> - {school}", edu_style))
                    
                    start_year = edu.get('startYear', '')
                    end_year = edu.get('endYear', '')
                    if start_year or end_year:
                        year_style = ParagraphStyle(
                            'EduYear',
                            parent=self.styles['Normal'],
                            fontSize=10,
                            textColor=colors.HexColor('#7f8c8d'),
                            spaceAfter=12
                        )
                        story.append(Paragraph(f"{start_year} - {end_year}", year_style))
        
        # Skills section
        skills = cv_data.get('skills', '')
        if skills:
            story.append(Paragraph("UMIEJĘTNOŚCI", self.section_header))
            skills_list = [skill.strip() for skill in skills.split(',') if skill.strip()]
            
            # Create skills in columns
            skills_data = []
            for i in range(0, len(skills_list), 3):
                row = skills_list[i:i+3]
                while len(row) < 3:
                    row.append("")
                skills_data.append([f"• {skill}" if skill else "" for skill in row])
            
            skills_table = Table(skills_data, colWidths=[doc.width/3]*3)
            skills_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2c3e50')),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]))
            story.append(skills_table)
        
        # Footer accent
        story.append(Spacer(1, 1*cm))
        story.append(ColorBox(doc.width, 0.3*cm, colors.HexColor('#3498db')))
        
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    def generate_creative_cv(self, cv_data):
        """Generate creative CV template with modern design"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=1.5*cm, leftMargin=1.5*cm, 
                               topMargin=1.5*cm, bottomMargin=1.5*cm)
        story = []
        
        # Creative header with gradient-like effect
        story.append(ColorBox(doc.width, 1*cm, colors.HexColor('#e74c3c')))
        story.append(Spacer(1, -0.8*cm))
        
        # Name in white on red background
        name = f"{cv_data.get('firstName', '')} {cv_data.get('lastName', '')}".strip()
        white_title = ParagraphStyle(
            'WhiteTitle',
            parent=self.creative_title,
            textColor=colors.white,
            alignment=1
        )
        story.append(Paragraph(name, white_title))
        story.append(Spacer(1, 0.3*cm))
        
        # Job title
        job_title = cv_data.get('jobTitle', '')
        if job_title:
            creative_subtitle = ParagraphStyle(
                'CreativeSubtitle',
                parent=self.styles['Heading2'],
                fontSize=14,
                textColor=colors.HexColor('#e74c3c'),
                spaceAfter=20,
                alignment=1,
                fontName='Helvetica-Oblique'
            )
            story.append(Paragraph(job_title, creative_subtitle))
        
        # Two-column layout for contact and content
        main_content = []
        
        # Contact sidebar
        contact_content = []
        contact_header = ParagraphStyle(
            'ContactHeader',
            parent=self.styles['Heading3'],
            fontSize=12,
            textColor=colors.HexColor('#e74c3c'),
            fontName='Helvetica-Bold',
            spaceAfter=10
        )
        
        contact_style = ParagraphStyle(
            'ContactStyle',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=5
        )
        
        contact_content.append(Paragraph("KONTAKT", contact_header))
        
        if cv_data.get('email'):
            contact_content.append(Paragraph(f"📧 {cv_data['email']}", contact_style))
        if cv_data.get('phone'):
            contact_content.append(Paragraph(f"📱 {cv_data['phone']}", contact_style))
        if cv_data.get('city'):
            contact_content.append(Paragraph(f"🏙️ {cv_data['city']}", contact_style))
        if cv_data.get('linkedin'):
            contact_content.append(Paragraph(f"💼 {cv_data['linkedin']}", contact_style))
        
        # Main content area
        if cv_data.get('summary'):
            creative_section = ParagraphStyle(
                'CreativeSection',
                parent=self.styles['Heading3'],
                fontSize=12,
                textColor=colors.HexColor('#e74c3c'),
                fontName='Helvetica-Bold',
                spaceAfter=10,
                spaceBefore=15
            )
            main_content.append(Paragraph("O MNIE", creative_section))
            main_content.append(Paragraph(cv_data['summary'], self.styles['Normal']))
        
        # Combine in table layout
        layout_data = []
        max_len = max(len(contact_content), len(main_content))
        
        for i in range(max_len):
            left = contact_content[i] if i < len(contact_content) else Spacer(1, 0)
            right = main_content[i] if i < len(main_content) else Spacer(1, 0)
            layout_data.append([left, right])
        
        layout_table = Table(layout_data, colWidths=[doc.width*0.3, doc.width*0.7])
        layout_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (0, -1), 0),
            ('RIGHTPADDING', (1, 0), (1, -1), 0),
        ]))
        story.append(layout_table)
        
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    def generate_executive_cv(self, cv_data):
        """Generate executive/corporate CV template"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=2.5*cm, leftMargin=2.5*cm, 
                               topMargin=2.5*cm, bottomMargin=2.5*cm)
        story = []
        
        # Executive header
        name = f"{cv_data.get('firstName', '')} {cv_data.get('lastName', '')}".strip()
        story.append(Paragraph(name, self.executive_title))
        
        # Elegant underline
        story.append(ColorBox(doc.width, 0.1*cm, colors.HexColor('#34495e')))
        story.append(Spacer(1, 0.5*cm))
        
        # Contact in elegant table
        if any([cv_data.get('email'), cv_data.get('phone'), cv_data.get('city')]):
            contact_data = [[
                cv_data.get('email', ''),
                cv_data.get('phone', ''),
                cv_data.get('city', '')
            ]]
            
            contact_table = Table(contact_data, colWidths=[doc.width/3]*3)
            contact_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, -1), 'Times-Roman'),
                ('FONTSIZE', (0, 0), (-1, -1), 11),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#34495e')),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
            ]))
            story.append(contact_table)
        
        # Professional sections with elegant styling
        exec_section = ParagraphStyle(
            'ExecSection',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#34495e'),
            fontName='Times-Bold',
            spaceAfter=12,
            spaceBefore=20,
            borderWidth=1,
            borderColor=colors.HexColor('#bdc3c7'),
            borderPadding=5
        )
        
        if cv_data.get('summary'):
            story.append(Paragraph("EXECUTIVE SUMMARY", exec_section))
            story.append(Paragraph(cv_data['summary'], self.styles['Normal']))
        
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    def generate_minimalist_cv(self, cv_data):
        """Generate clean minimalist CV template"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=3*cm, leftMargin=3*cm, 
                               topMargin=3*cm, bottomMargin=3*cm)
        story = []
        
        # Minimalist header
        name = f"{cv_data.get('firstName', '')} {cv_data.get('lastName', '')}".strip()
        story.append(Paragraph(name, self.minimal_title))
        
        # Simple line
        story.append(ColorBox(doc.width, 0.05*cm, colors.black))
        story.append(Spacer(1, 1*cm))
        
        # Clean typography throughout
        minimal_section = ParagraphStyle(
            'MinimalSection',
            parent=self.styles['Heading3'],
            fontSize=12,
            textColor=colors.black,
            fontName='Helvetica',
            spaceAfter=15,
            spaceBefore=25,
            leftIndent=0
        )
        
        # Content with lots of white space
        if cv_data.get('summary'):
            story.append(Paragraph("About", minimal_section))
            story.append(Paragraph(cv_data['summary'], self.styles['Normal']))
        
        doc.build(story)
        buffer.seek(0)
        return buffer

def generate_cv_with_template(cv_data, template_style="modern_blue"):
    """Main function to generate CV with selected template"""
    generator = CVTemplateGenerator()
    
    if template_style == "modern_blue":
        return generator.generate_modern_blue_cv(cv_data)
    elif template_style == "creative":
        return generator.generate_creative_cv(cv_data)
    elif template_style == "executive":
        return generator.generate_executive_cv(cv_data)
    elif template_style == "minimalist":
        return generator.generate_minimalist_cv(cv_data)
    else:
        # Default to modern blue
        return generator.generate_modern_blue_cv(cv_data)
import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def generate_cv_with_template(cv_data, template_style='modern_blue'):
    """
    Generate CV PDF with selected template style
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Template-specific styling
    if template_style == 'modern_blue':
        primary_color = colors.HexColor('#3498db')
        secondary_color = colors.HexColor('#2c3e50')
    elif template_style == 'creative':
        primary_color = colors.HexColor('#e74c3c')
        secondary_color = colors.HexColor('#f39c12')
    elif template_style == 'executive':
        primary_color = colors.HexColor('#34495e')
        secondary_color = colors.HexColor('#2c3e50')
    else:  # minimalist
        primary_color = colors.HexColor('#2c3e50')
        secondary_color = colors.HexColor('#95a5a6')

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=primary_color,
        spaceAfter=30,
        alignment=1  # Center
    )

    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=secondary_color,
        spaceAfter=20
    )

    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=12
    )

    # Header
    name = f"{cv_data.get('firstName', '')} {cv_data.get('lastName', '')}".strip()
    story.append(Paragraph(name, title_style))

    job_title = cv_data.get('jobTitle', '')
    if job_title:
        story.append(Paragraph(job_title, styles['Heading3']))

    # Contact info
    contact_info = []
    if cv_data.get('email'):
        contact_info.append(cv_data['email'])
    if cv_data.get('phone'):
        contact_info.append(cv_data['phone'])
    if cv_data.get('city'):
        contact_info.append(cv_data['city'])

    if contact_info:
        story.append(Paragraph(' | '.join(contact_info), normal_style))

    story.append(Spacer(1, 20))

    # Summary
    if cv_data.get('summary'):
        story.append(Paragraph("O mnie", subtitle_style))
        story.append(Paragraph(cv_data['summary'], normal_style))
        story.append(Spacer(1, 15))

    # Experience
    experiences = cv_data.get('experiences', [])
    if experiences and any(exp.get('title') or exp.get('company') for exp in experiences):
        story.append(Paragraph("Doświadczenie zawodowe", subtitle_style))
        for exp in experiences:
            if exp.get('title') or exp.get('company'):
                # Title and company
                exp_header = f"<b>{exp.get('title', 'Stanowisko')}</b> - {exp.get('company', 'Firma')}"
                story.append(Paragraph(exp_header, normal_style))

                # Dates
                start_date = exp.get('startDate', '')
                end_date = exp.get('endDate', 'obecnie')
                if start_date:
                    date_range = f"{start_date} - {end_date}"
                    story.append(Paragraph(date_range, normal_style))

                # Description
                if exp.get('description'):
                    story.append(Paragraph(exp['description'], normal_style))

                story.append(Spacer(1, 10))

    # Education
    education = cv_data.get('education', [])
    if education and any(edu.get('degree') or edu.get('school') for edu in education):
        story.append(Paragraph("Wykształcenie", subtitle_style))
        for edu in education:
            if edu.get('degree') or edu.get('school'):
                # Degree and school
                edu_header = f"<b>{edu.get('degree', 'Kierunek')}</b> - {edu.get('school', 'Uczelnia')}"
                story.append(Paragraph(edu_header, normal_style))

                # Years
                start_year = edu.get('startYear', '')
                end_year = edu.get('endYear', '')
                if start_year or end_year:
                    year_range = f"{start_year} - {end_year}"
                    story.append(Paragraph(year_range, normal_style))

                story.append(Spacer(1, 10))

    # Skills
    skills = cv_data.get('skills', '')
    if skills:
        story.append(Paragraph("Umiejętności", subtitle_style))
        skills_list = [skill.strip() for skill in skills.split(',') if skill.strip()]
        skills_text = ' • '.join(skills_list)
        story.append(Paragraph(skills_text, normal_style))

    doc.build(story)
    buffer.seek(0)
    return buffer
