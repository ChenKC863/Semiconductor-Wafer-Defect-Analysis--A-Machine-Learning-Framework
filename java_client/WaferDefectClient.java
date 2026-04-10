import java.net.URL;
import java.net.HttpURLConnection;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.io.IOException;
import java.nio.file.Files;
import java.io.OutputStream;
import java.io.BufferedReader;
import java.io.InputStreamReader;

public class WaferDefectClient {
    public static void main(String[] args) throws Exception {
        String imagePath = "test.jpg";
        String apiUrl = "http://127.0.0.1:8000/predict";
        if (args.length >= 1) imagePath = args[0];
        if (args.length >= 2) apiUrl = args[1];

        Path filePath = Paths.get(imagePath);
        if (!Files.exists(filePath)) {
            System.err.println("檔案不存在: " + filePath.toAbsolutePath());
            return;
        }

        String boundary = "JavaBoundary1234567890";
        String CRLF = "\r\n";
        String mimeType = Files.probeContentType(filePath);
        if (mimeType == null) {
            mimeType = "application/octet-stream";
        }

        URL url = new URL(apiUrl);
        HttpURLConnection connection = (HttpURLConnection) url.openConnection();
        connection.setRequestMethod("POST");
        connection.setDoOutput(true);
        connection.setRequestProperty("Content-Type", "multipart/form-data; boundary=" + boundary);
        connection.setRequestProperty("Accept", "application/json");

        try (OutputStream output = connection.getOutputStream()) {
            StringBuilder headerBuilder = new StringBuilder();
            headerBuilder.append("--").append(boundary).append(CRLF);
            headerBuilder.append("Content-Disposition: form-data; name=\"file\"; filename=\"")
                    .append(filePath.getFileName()).append("\"").append(CRLF);
            headerBuilder.append("Content-Type: ").append(mimeType).append(CRLF);
            headerBuilder.append(CRLF);
            output.write(headerBuilder.toString().getBytes(StandardCharsets.UTF_8));
            output.write(Files.readAllBytes(filePath));
            output.write(CRLF.getBytes(StandardCharsets.UTF_8));
            output.write(("--" + boundary + "--" + CRLF).getBytes(StandardCharsets.UTF_8));
            output.flush();
        }

        int status = connection.getResponseCode();
        BufferedReader reader = new BufferedReader(new InputStreamReader(
                status >= 400 ? connection.getErrorStream() : connection.getInputStream(), StandardCharsets.UTF_8));

        StringBuilder responseBody = new StringBuilder();
        String line;
        while ((line = reader.readLine()) != null) {
            responseBody.append(line);
        }
        reader.close();

        System.out.println("Status code: " + status);
        System.out.println("Response: " + responseBody.toString());
    }

}
