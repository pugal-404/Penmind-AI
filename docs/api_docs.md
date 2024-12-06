# API Documentation for Handwriting Recognition System

## Base URL

All API requests should be sent to: `http://localhost:8000`

## Endpoints

### Recognize Handwriting

Endpoint: `/recognize`
Method: POST
Content-Type: multipart/form-data

#### Request

| Parameter | Type | Description |
|-----------|------|-------------|
| file      | File | The image file containing handwritten text |

#### Response

```json
{
  "text": "Recognized text from the image"
}