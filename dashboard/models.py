from django.db import models
from django.utils import timezone

class Prediction(models.Model):
    PREDICTION_TYPES = [
        ('flood', 'Flood'),
        ('drought', 'Drought'),
    ]
    
    FORECAST_PERIODS = [
        (0, 'Current Day'),
        (7, '7 Days Ahead'),
        (14, '14 Days Ahead'),
        (21, '21 Days Ahead'),
    ]
    
    prediction_type = models.CharField(max_length=10, choices=PREDICTION_TYPES)
    forecast_period = models.IntegerField(choices=FORECAST_PERIODS)
    prediction_date = models.DateField()
    created_at = models.DateTimeField(default=timezone.now)
    
    # Store GEE image metadata
    gee_image_id = models.CharField(max_length=255, blank=True, null=True)
    tile_url = models.URLField(blank=True, null=True)
    
    # Summary statistics
    affected_area_km2 = models.FloatField(blank=True, null=True)
    severity_level = models.CharField(max_length=20, blank=True, null=True)
    
    class Meta:
        unique_together = ['prediction_type', 'forecast_period', 'prediction_date']
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.get_prediction_type_display()} - {self.get_forecast_period_display()} - {self.prediction_date}"

class PredictionMetadata(models.Model):
    """Store additional metadata for predictions"""
    prediction = models.OneToOneField(Prediction, on_delete=models.CASCADE, related_name='metadata')
    
    # Model performance metrics
    accuracy = models.FloatField(blank=True, null=True)
    confidence_score = models.FloatField(blank=True, null=True)
    
    # Processing information
    processing_time_seconds = models.FloatField(blank=True, null=True)
    data_sources = models.TextField(blank=True, null=True)  # JSON string of data sources used
    
    # Visualization parameters
    color_palette = models.CharField(max_length=100, blank=True, null=True)
    min_value = models.FloatField(blank=True, null=True)
    max_value = models.FloatField(blank=True, null=True)
    
    def __str__(self):
        return f"Metadata for {self.prediction}"

