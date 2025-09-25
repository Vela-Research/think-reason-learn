document.addEventListener('DOMContentLoaded', function() {
    if (typeof DOCUMENTATION_OPTIONS !== 'undefined' && DOCUMENTATION_OPTIONS.pagename) {
        const pagename = DOCUMENTATION_OPTIONS.pagename;
        if (pagename === 'modules' || pagename.startsWith('think_reason_learn')) {
            document.body.setAttribute('data-api-page', 'true');
        }
    }
});
