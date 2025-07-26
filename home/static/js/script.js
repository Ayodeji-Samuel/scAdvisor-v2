const icon = document.querySelector(".nav__icon");
const sidebar = document.querySelector(".sidebar__overlay");
const close = document.querySelector(".close__icon");
const errors = document.querySelector(".errorslist");


icon.addEventListener("click", () => {
  sidebar.classList.toggle("open");
});

close.addEventListener("click", () => {
  sidebar.classList.toggle("open");
});


setTimeout(() =>{

  errors
}, 3000)