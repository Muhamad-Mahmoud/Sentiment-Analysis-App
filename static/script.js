// ================================================
// Sentiment Analysis App - Interactive Features
// ================================================

console.log('✨ Sentiment Analysis App loaded successfully!');

// Smooth scroll for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        const href = this.getAttribute('href');
        if (href.startsWith('#') && href.length > 1) {
            e.preventDefault();
            const targetElement = document.querySelector(href);
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        } else if (href === '#') {
            e.preventDefault();
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        }
    });
});

// Add active class to nav links based on current URL
const currentPath = window.location.pathname;
const navLinks = document.querySelectorAll('.nav-link');

navLinks.forEach(link => {
    const href = link.getAttribute('href');
    if (href === currentPath) {
        link.classList.add('active');
    } else {
        link.classList.remove('active');
    }
});