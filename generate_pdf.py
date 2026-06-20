"""Generate PDF from the HTML presentation using Playwright."""
import os
from pathlib import Path
from playwright.sync_api import sync_playwright

def main():
    html_path = Path(__file__).parent / "presentation.html"
    pdf_path = Path(__file__).parent / "AML_Risk_Scoring_Presentation.pdf"
    
    file_url = html_path.as_uri()
    
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        
        # Load the HTML file
        page.goto(file_url, wait_until="networkidle")
        
        # Make all slides visible for PDF generation
        page.evaluate("""() => {
            // Remove navigation controls
            document.querySelector('.nav-controls')?.remove();
            document.querySelector('.slide-number')?.remove();
            
            // Show all slides and style them for print
            const slides = document.querySelectorAll('.slide');
            slides.forEach(slide => {
                slide.style.display = 'flex';
                slide.style.flexDirection = 'column';
                slide.style.width = '100%';
                slide.style.height = '100vh';
                slide.style.pageBreakAfter = 'always';
                slide.style.overflow = 'hidden';
                slide.style.padding = '40px 60px';
            });
            
            // Remove the active class behavior
            document.body.style.overflow = 'visible';
        }""")
        
        # Wait for images to load
        page.wait_for_timeout(3000)
        
        # Generate PDF - landscape A4
        page.pdf(
            path=str(pdf_path),
            format="A4",
            landscape=True,
            print_background=True,
            margin={"top": "0", "bottom": "0", "left": "0", "right": "0"}
        )
        
        browser.close()
    
    print(f"PDF generated: {pdf_path}")
    print(f"File size: {pdf_path.stat().st_size / 1024:.1f} KB")

if __name__ == "__main__":
    main()
