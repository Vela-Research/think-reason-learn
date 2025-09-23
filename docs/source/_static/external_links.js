// Make external links open in new tabs
document.addEventListener('DOMContentLoaded', function() {
    // Get all links on the page
    const links = document.querySelectorAll('a[href^="http"]');
    
    links.forEach(function(link) {
        const href = link.getAttribute('href');
        
        // Check if it's an external link (not localhost or current domain)
        if (href && 
            !href.includes('localhost') && 
            !href.includes('127.0.0.1') &&
            !href.includes(window.location.hostname)) {
            
            // Add target="_blank" and rel="noopener noreferrer" for security
            link.setAttribute('target', '_blank');
            link.setAttribute('rel', 'noopener noreferrer');
            
            // Add a title attribute to indicate it opens in new tab
            const currentTitle = link.getAttribute('title') || '';
            const newTabText = ' (opens in new tab)';
            if (!currentTitle.includes(newTabText)) {
                link.setAttribute('title', currentTitle + newTabText);
            }
        }
    });
});
