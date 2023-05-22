from transformers import AutoTokenizer, AutoModelForSequenceClassification


def main():
    model_name = 'bert-base-cased' 
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    while True:
        text = input("Enter text (or type 'q' to quit): ")
        if text == 'q':
            break

        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        outputs = model(**inputs)
        predicted_class = outputs.logits.argmax().item()
        print(outputs.logits)

        if predicted_class == 0:
            print("Hate speech")
        elif predicted_class == 1:
            print("Offensive")
        else:
            print("Normal")

if __name__ == "__main__":
    main()