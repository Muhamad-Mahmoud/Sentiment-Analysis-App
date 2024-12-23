// Select the buttons and sections
const toggleTeamButton = document.getElementById('toggle-team');
const toggleProjectButton = document.getElementById('toggle-project');
const teamInfoSection = document.getElementById('team-info');
const projectInfoSection = document.getElementById('project-info');
const loadingBar = document.getElementById('loading-bar');
const cleanedText = document.querySelector('.cleaned_text'); // Reference the textarea

// Show progress bar and simulate loading
document.querySelector('form').addEventListener('submit', (event) => {
    loadingBar.classList.remove('hidden');
    const progressBar = loadingBar.querySelector('.progress-bar');
    progressBar.style.width = '0%';
    setTimeout(() => {
        progressBar.style.width = '100%';
    }, 300); // Simulate a 300ms delay
});



toggleTeamButton.addEventListener('click', () => {
    const isHidden = teamInfoSection.classList.contains('hidden');
    teamInfoSection.classList.toggle('hidden', !isHidden);
    projectInfoSection.classList.add('hidden'); // Hide the other section
    toggleTeamButton.setAttribute('aria-expanded', isHidden);
});

// Toggle visibility for the "Project Information" section
toggleProjectButton.addEventListener('click', () => {
    const isHidden = projectInfoSection.classList.contains('hidden');
    projectInfoSection.classList.toggle('hidden', !isHidden);
    teamInfoSection.classList.add('hidden'); // Hide the other section
    toggleProjectButton.setAttribute('aria-expanded', isHidden);
    cleaned_text.style.display='block'
});

submitBtn.addEventListener('click', (event) => {
    const isHidden = cleaned_text.classList.contains('hidden');
    cleaned_text.classList.toggle('hidden', !isHidden);
    cleaned_text.style.display='block'
});