//! OpenRouter client tests: wiremock-canned `chat/completions` responses,
//! payload decode, and error mapping.

use serde_json::{Value, json};
use stop_dataset::error::DatasetError;
use stop_dataset::openrouter::OpenRouterClient;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn completion(content: Value) -> Value {
    json!({
        "choices": [
            { "message": { "role": "assistant", "content": content } }
        ]
    })
}

#[tokio::test]
async fn decodes_chat_completion_content_as_json() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(completion(json!("{\"a\": 1}"))))
        .expect(1)
        .mount(&server)
        .await;
    let client = OpenRouterClient::new(server.uri(), "test-key", "test-model");

    let value = client.complete_json("system", "user").await.expect("json");
    assert_eq!(value["a"], 1);

    let requests = server.received_requests().await.expect("request log");
    let body: Value = serde_json::from_slice(&requests[0].body).expect("request body is JSON");
    assert_eq!(body["model"], json!("test-model"));
    assert_eq!(body["messages"][0]["content"], json!("system"));
    assert_eq!(body["messages"][1]["content"], json!("user"));
    assert_eq!(body["response_format"]["type"], json!("json_object"));
}

#[tokio::test]
async fn http_error_status_maps_to_dataset_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(401).set_body_string("unauthorized"))
        .mount(&server)
        .await;
    let client = OpenRouterClient::new(server.uri(), "test-key", "test-model");

    let error = client
        .complete_json("system", "user")
        .await
        .expect_err("401");

    match error {
        DatasetError::Status { status, body } => {
            assert_eq!(status, 401);
            assert!(body.contains("unauthorized"), "body: {body}");
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[tokio::test]
async fn non_json_content_maps_to_malformed() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(completion(json!("I cannot help"))))
        .mount(&server)
        .await;
    let client = OpenRouterClient::new(server.uri(), "test-key", "test-model");

    let error = client
        .complete_json("system", "user")
        .await
        .expect_err("no json");

    assert!(matches!(error, DatasetError::Malformed(_)), "got {error:?}");
}

#[tokio::test]
async fn missing_choices_maps_to_malformed() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
        .mount(&server)
        .await;
    let client = OpenRouterClient::new(server.uri(), "test-key", "test-model");

    let error = client
        .complete_json("system", "user")
        .await
        .expect_err("empty response");

    assert!(matches!(error, DatasetError::Malformed(_)), "got {error:?}");
}
