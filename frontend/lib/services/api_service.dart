import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';

class ApiService {
  // Change this when deploying to VM
  static String baseUrl = 'http://localhost:8000';

  /// Get MIME type from filename extension
  static String _getMimeType(String filename) {
    final ext = filename.split('.').last.toLowerCase();
    switch (ext) {
      case 'jpg':
      case 'jpeg':
        return 'image/jpeg';
      case 'png':
        return 'image/png';
      case 'gif':
        return 'image/gif';
      case 'webp':
        return 'image/webp';
      case 'bmp':
        return 'image/bmp';
      case 'mp4':
        return 'video/mp4';
      case 'avi':
        return 'video/x-msvideo';
      case 'mov':
        return 'video/quicktime';
      case 'webm':
        return 'video/webm';
      case 'mkv':
        return 'video/x-matroska';
      case 'wav':
        return 'audio/wav';
      case 'mp3':
        return 'audio/mpeg';
      case 'm4a':
        return 'audio/mp4';
      case 'aac':
        return 'audio/aac';
      case 'ogg':
        return 'audio/ogg';
      case 'flac':
        return 'audio/flac';
      default:
        return 'application/octet-stream';
    }
  }

  static Future<Map<String, dynamic>> predictAudio(
    Uint8List bytes,
    String filename,
  ) async {
    final uri = Uri.parse('$baseUrl/predict-audio');
    final request = http.MultipartRequest('POST', uri);
    final contentType = _getMimeType(filename);
    request.files.add(
      http.MultipartFile.fromBytes(
        'file',
        bytes,
        filename: filename,
        contentType: MediaType.parse(contentType),
      ),
    );

    final streamedResponse = await request.send();
    final response = await http.Response.fromStream(streamedResponse);

    if (response.statusCode == 200) {
      return jsonDecode(response.body) as Map<String, dynamic>;
    } else {
      final body = response.body.isNotEmpty ? jsonDecode(response.body) : null;
      final detail = body is Map ? body['detail'] : null;
      final msg = detail is String ? detail : (detail is List && detail.isNotEmpty && detail[0] is Map ? (detail[0] as Map)['msg']?.toString() : null);
      throw Exception(msg ?? 'Failed to analyze audio');
    }
  }

  static Future<Map<String, dynamic>> predictImage(
    Uint8List bytes,
    String filename,
  ) async {
    final uri = Uri.parse('$baseUrl/predict');
    final request = http.MultipartRequest('POST', uri);
    final contentType = _getMimeType(filename);
    request.files.add(
      http.MultipartFile.fromBytes(
        'file',
        bytes,
        filename: filename,
        contentType: MediaType.parse(contentType),
      ),
    );

    final streamedResponse = await request.send();
    final response = await http.Response.fromStream(streamedResponse);

    if (response.statusCode == 200) {
      return jsonDecode(response.body) as Map<String, dynamic>;
    } else {
      final body = response.body.isNotEmpty ? jsonDecode(response.body) : null;
      final detail = body is Map ? body['detail'] : null;
      final msg = detail is String ? detail : (detail is List && detail.isNotEmpty && detail[0] is Map ? (detail[0] as Map)['msg']?.toString() : null);
      throw Exception(msg ?? 'Failed to analyze image');
    }
  }

  static Future<Map<String, dynamic>> predictVideo(
    Uint8List bytes,
    String filename,
  ) async {
    final uri = Uri.parse('$baseUrl/predict-video');
    final request = http.MultipartRequest('POST', uri);
    final contentType = _getMimeType(filename);
    request.files.add(
      http.MultipartFile.fromBytes(
        'file',
        bytes,
        filename: filename,
        contentType: MediaType.parse(contentType),
      ),
    );

    final streamedResponse = await request.send();
    final response = await http.Response.fromStream(streamedResponse);

    if (response.statusCode == 200) {
      return jsonDecode(response.body) as Map<String, dynamic>;
    } else {
      final body = response.body.isNotEmpty ? jsonDecode(response.body) : null;
      final detail = body is Map ? body['detail'] : null;
      final msg = detail is String ? detail : (detail is List && detail.isNotEmpty && detail[0] is Map ? (detail[0] as Map)['msg']?.toString() : null);
      throw Exception(msg ?? 'Failed to analyze video');
    }
  }
}
