const icon = document.querySelector(".nav__icon");
const sidebar = document.querySelector(".sidebar__overlay");
const close = document.querySelector(".close__icon");

// Sidebar toggle
icon.addEventListener("click", () => {
  sidebar.classList.toggle("open");
});

close.addEventListener("click", () => {
  sidebar.classList.toggle("open");
});

// Close sidebar when clicking overlay area
sidebar.addEventListener("click", (e) => {
  if (e.target === sidebar) {
    sidebar.classList.remove("open");
  }
});

// Navbar scroll effect
const navbar = document.getElementById("mainNav");
if (navbar) {
  window.addEventListener("scroll", () => {
    if (window.scrollY > 50) {
      navbar.classList.add("scrolled");
    } else {
      navbar.classList.remove("scrolled");
    }
  });
}

// Auto-dismiss alert messages
document.querySelectorAll(".alert").forEach((alert) => {
  setTimeout(() => {
    alert.style.transition = "opacity 0.5s ease, transform 0.5s ease";
    alert.style.opacity = "0";
    alert.style.transform = "translateX(100%)";
    setTimeout(() => alert.remove(), 500);
  }, 5000);
});