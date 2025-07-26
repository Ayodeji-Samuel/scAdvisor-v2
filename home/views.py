from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.views import LoginView
from django.contrib import messages
from django.urls import reverse_lazy
from django.views.generic import CreateView
from django.contrib.auth.models import User

# Create your views here.

def index(request):
    """Home page view"""
    return render(request, 'home/index.html')

def about(request):
    """About page view"""
    return render(request, 'home/about.html')

def contact(request):
    """Contact page view"""
    return render(request, 'home/contact.html')

class CustomUserCreationForm(UserCreationForm):
    """Custom user creation form with additional fields"""
    class Meta:
        model = User
        fields = ('username', 'email', 'password1', 'password2')
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['username'].widget.attrs.update({'class': 'form-control', 'placeholder': 'Username'})
        self.fields['email'].widget.attrs.update({'class': 'form-control', 'placeholder': 'Email'})
        self.fields['password1'].widget.attrs.update({'class': 'form-control', 'placeholder': 'Password'})
        self.fields['password2'].widget.attrs.update({'class': 'form-control', 'placeholder': 'Confirm Password'})

class SignUpView(CreateView):
    """User registration view"""
    form_class = CustomUserCreationForm
    template_name = 'registration/signup.html'
    success_url = reverse_lazy('login')

    def form_valid(self, form):
        response = super().form_valid(form)
        messages.success(self.request, 'Account created successfully! Please log in.')
        return response

class CustomLoginView(LoginView):
    """Custom login view"""
    template_name = 'registration/login.html'
    
    def form_valid(self, form):
        messages.success(self.request, 'Welcome back!')
        return super().form_valid(form)

def logout_view(request):
    """Logout the user and redirect to the home page"""
    logout(request)
    messages.success(request, 'You have been logged out successfully.')
    return redirect('index')
