// ================================================
// Sentiment Analysis App - Interactive Features
// ================================================

// Get all interactive elements
const toggleTeamButton = document.getElementById('toggle-team');
const toggleProjectButton = document.getElementById('toggle-project');
const teamInfoSection = document.getElementById('team-info');
const projectInfoSection = document.getElementById('project-info');

// Navigation links
const navTeamLink = document.getElementById('nav-team');
const navAboutLink = document.getElementById('nav-about');
const footerTeamLinks = document.querySelectorAll('.footer-team-link');
const footerAboutLinks = document.querySelectorAll('.footer-about-link');

// Toggle Team Section
function toggleTeamSection() {
    const isHidden = teamInfoSection.classList.contains('hidden');

    // Toggle team section
    teamInfoSection.classList.toggle('hidden');
    projectInfoSection.classList.add('hidden');

    // Scroll to section if opening
    if (isHidden) {
        setTimeout(() => {
            teamInfoSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 100);
    }
}

// Toggle Project Section
function toggleProjectSection() {
    const isHidden = projectInfoSection.classList.contains('hidden');

    // Toggle project section
    projectInfoSection.classList.toggle('hidden');
    teamInfoSection.classList.add('hidden');

    // Scroll to section if opening
    if (isHidden) {
        setTimeout(() => {
            projectInfoSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 100);
    }
}

// Add event listeners for toggle buttons
if (toggleTeamButton) {
    toggleTeamButton.addEventListener('click', toggleTeamSection);
}

if (toggleProjectButton) {
    toggleProjectButton.addEventListener('click', toggleProjectSection);
}

// Add event listeners for navigation links
if (navTeamLink) {
    navTeamLink.addEventListener('click', (e) => {
        e.preventDefault();
        toggleTeamSection();
    });
}

if (navAboutLink) {
    navAboutLink.addEventListener('click', (e) => {
        e.preventDefault();
        toggleProjectSection();
    });
}

// Add event listeners for footer links
footerTeamLinks.forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        toggleTeamSection();
        // Scroll to top after a brief delay
        setTimeout(() => {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }, 300);
    });
});

footerAboutLinks.forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        toggleProjectSection();
        // Scroll to top after a brief delay
        setTimeout(() => {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }, 300);
    });
});

// Smooth scroll for navigation
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        const href = this.getAttribute('href');
        if (href === '#home' || href === '#') {
            e.preventDefault();
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        }
    });
});

// Add active class to nav links on scroll
const navLinks = document.querySelectorAll('.nav-link');
window.addEventListener('scroll', () => {
    const scrollPos = window.scrollY;

    navLinks.forEach(link => {
        if (link.getAttribute('href') === '#home') {
            if (scrollPos < 100) {
                link.classList.add('active');
            } else {
                link.classList.remove('active');
            }
        }
    });
});

// Animate elements on scroll (fade in effect)
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Observe sections for animation
document.querySelectorAll('.info-section, .hero-header, .analysis-card').forEach(section => {
    section.style.opacity = '0';
    section.style.transform = 'translateY(20px)';
    section.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    observer.observe(section);
});

console.log('✨ Sentiment Analysis App loaded successfully!');