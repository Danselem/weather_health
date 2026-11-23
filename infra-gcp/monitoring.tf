resource "google_monitoring_notification_channel" "email" {
  display_name = "Weather Health email alerts"
  type         = "email"
  labels = {
    email_address = var.alert_email
  }
}

resource "google_monitoring_uptime_check_config" "health" {
  display_name = "weather-health /health"
  timeout      = "10s"
  period       = "60s"

  http_check {
    path         = "/health"
    port         = 443
    use_ssl      = true
    validate_ssl = true
  }

  monitored_resource {
    type = "uptime_url"
    labels = {
      project_id = var.project_id
      host       = replace(module.cloudrun.service_url, "https://", "")
    }
  }
}

resource "google_monitoring_alert_policy" "uptime" {
  display_name = "weather-health uptime"
  combiner     = "OR"

  conditions {
    display_name = "health check failing"

    condition_threshold {
      filter          = "metric.type=\"monitoring.googleapis.com/uptime_check/check_passed\" AND resource.type=\"uptime_url\" AND metric.label.check_id=\"${google_monitoring_uptime_check_config.health.uptime_check_id}\""
      duration        = "120s"
      comparison      = "COMPARISON_GT"
      threshold_value = 1

      trigger {
        count = 1
      }

      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_NEXT_OLDER"
        cross_series_reducer = "REDUCE_COUNT_FALSE"
        group_by_fields      = ["resource.label.*"]
      }
    }
  }

  notification_channels = [google_monitoring_notification_channel.email.id]
}

resource "google_logging_metric" "drift_failure" {
  name        = "weather-health-drift-failures"
  description = "Counts monitoring jobs that fail the drift gate"
  filter      = "resource.type=\"cloud_run_job\" AND resource.labels.job_name=\"weather-health-monitoring\" AND textPayload:\"Drift gate failed\""

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    unit        = "1"
  }
}

resource "google_monitoring_alert_policy" "drift" {
  display_name = "weather-health drift failure"
  combiner     = "OR"

  conditions {
    display_name = "drift gate failed"

    condition_threshold {
      filter          = "metric.type=\"logging.googleapis.com/user/weather-health-drift-failures\" AND resource.type=\"cloud_run_job\""
      comparison      = "COMPARISON_GT"
      threshold_value = 0
      duration        = "0s"

      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_DELTA"
        cross_series_reducer = "REDUCE_SUM"
      }
    }
  }

  notification_channels = [google_monitoring_notification_channel.email.id]
}
