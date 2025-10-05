// Debug script to test theme functionality
console.log('Testing theme functionality...');

// Check if dark class is applied
console.log('HTML classes:', document.documentElement.classList);

// Test toggling
function testToggle() {
  console.log('Toggling theme...');
  const isDark = document.documentElement.classList.toggle('dark');
  console.log('Dark mode is now:', isDark);
  console.log('HTML classes after toggle:', document.documentElement.classList);
  return isDark;
}

// Test localStorage
function testStorage() {
  console.log('Current theme in localStorage:', localStorage.getItem('theme'));
  localStorage.setItem('theme', 'dark');
  console.log('Set theme to dark in localStorage');
  console.log('Theme in localStorage:', localStorage.getItem('theme'));
}

console.log('Available functions:');
console.log('- testToggle(): Toggle dark mode');
console.log('- testStorage(): Test localStorage functionality');