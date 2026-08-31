import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_NAME = "ProsusAI/finbert"


def resolver_dispositivo(requested="auto"):
    """Resolve auto/cuda/cpu and fall back to CPU when CUDA is unavailable."""
    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError("device must be auto, cuda, or cpu")
    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if requested == "cuda":
        print("CUDA no está disponible; usando CPU.")
    return "cpu"


def cargar_modelo_finbert(model_name=MODEL_NAME, device="auto"):
    """Carga FinBERT solo cuando el pipeline realmente necesita inferencia."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = resolver_dispositivo(device)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def analizar_sentimiento(titulares, tokenizer=None, model=None, device=None):
    """
    Recibe una lista de titulares y devuelve el sentimiento y los scores.
    """
    if tokenizer is None or model is None or device is None:
        tokenizer, model, device = cargar_modelo_finbert()

    # Tokenizar las entradas
    inputs = tokenizer(titulares, padding=True, truncation=True, return_tensors="pt").to(device)
    
    # Inferencia (desactivamos gradientes para ahorrar memoria)
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Aplicar Softmax para obtener probabilidades
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    etiquetas = ["positive", "negative", "neutral"]
    resultados = []
    
    for i, probs in enumerate(predictions):
        prob_dict = {etiquetas[j]: probs[j].item() for j in range(len(etiquetas))}
        sentimiento_dominante = etiquetas[torch.argmax(probs).item()]
        
        resultados.append({
            "texto": titulares[i],
            "sentiment": sentimiento_dominante,
            "score_positive": prob_dict["positive"],
            "score_negative": prob_dict["negative"],
            "score_neutral": prob_dict["neutral"]
        })
        
    return resultados

if __name__ == "__main__":
    # --- PRUEBA CON NOTICIAS ---
    noticias = [
    "Tesla Recalls 675,000 Cars in US, China Tesla Recalls 675,000 Cars in US, China Voice of America 01 Jan 2022, 04:05 GMT+10 PARIS, FRANCE - Tesla has recalled 675,000 cars in the United States and China over issues with the trunk and front hood of two models, raising new questions about the safety of the popular electric vehicle. Chinese regulators announced the recall of almost 200,000 cars on Friday, hours after some 475,000 Tesla vehicles were flagged in the United States. The problems with the trunk and hood increase the risk of crashes, according to U.S. and Chinese regulators. Authorities said the repeated opening and closing of the trunk of the Model 3 can damage a cable for the rearview camera. An issue with the latchassembly for the front hood of the Model S could cause it to open without warning and obstruct the driver's visibility, according to the U.S. National Highway Traffic Safety Administration (NHTSA). Tesla estimates that the problems affect 1% of Model 3 and 14% of Model S vehicles recalled in the United States, without causing any accidents so far. Mass recalls are not rare in the auto industry. Volkswagen had to take 8.5 million cars out of circulation in 2015 due to the Dieselgate scandal, in which the German company admitted tampering with millions of diesel vehicles to dupe emissions tests. At least 100 million vehicles were recalled by car companies across the world in recent years due to a defect with airbags made by bankrupt Japanese group Takata. Tesla's recall represents a quarter of the number of cars Elon Musk's young company has produced so far. 'It is a reality wake-up call for Tesla though, with a slap-in-the-face welcome to the automotive world that is perhaps more complex than the smartphone industry that many like to compare it to,' said German auto analyst Matthias Schmidt. 'After all, a dysfunctional car on four wheels can do a lot more potential damage than a dysfunctional iPhone,' Schmidt said. Other incidents In June, Tesla recalled more than 285,000 cars in China over issues with its assisted driving software that could cause accidents. The company also recalled thousands of Model 3 and Model Y vehicles earlier that month to inspect brake calipers for loose bolts. In November, the NHTSA recalled nearly 12,000 Tesla cars due to errors with their communication software. U.S. safety officials are also investigating Tesla's Autopilot after identifying 11 crashes involving the driver assistance system. The previous month, U.S. highway safety regulators demanded details from Tesla on issues with its new autonomous system, building on a previously announced probe. Tesla executives have downplayed the regulatory inquiries, saying they were to be expected with 'cutting edge' technology and that they were cooperating 'as much as possible.' Banner year The issues have been blights to an otherwise banner year for Tesla, as it joined the exclusive club of companies with a market capitalization of $1 trillion. The company delivered a record 240,000 vehicles in the third quarter, and Tesla's billionaire chief Elon Musk was named Time magazine's person of the year. Tesla's good fortune contrasted with other, traditional automakers that were heavily affected by the coronavirus pandemic and a shortage of semiconductors that are key components in cars. Trip Chowdhry, analyst at Global Equities Research consultancy, said the latest Tesla recall is a 'non-event' as the company still holds a big advantage over its competitors. Share article: Shares Share Tweet Share Flip Email Watch latest videos Subscribe and Follow Get a daily dose of Singapore Star news"
    "through our daily email, its complimentary and keeps you fully"
    "up to date with world and business news as well. Subscribe Now News RELEASES Publish news of your business, community or sports group,"
    "personnel appointments, major event and more by submitting a"
    "news release to Singapore Star . More Information null in null International Section Boeing under scrutiny after deadly Dreamliner crash in India SEATTLE/BENGALURU: Boeing is once again under scrutiny following the crash of an Air India 787-8 Dreamliner that killed nearly all... Brian Wilson, legendary Beach Boys cofounder, dies at 82 LOS ANGELES, California: Brian Wilson, the musical genius behind many of the Beach Boys' greatest hits like Good Vibrations and God... UN nuclear authority 'deeply concerned' about attacks on Iran's nuclear plants NEW YORK, New York - The head of the International Atomic Energy Agency (IAEA) says it is deeply concerning that Israel is carrying... Sirens sound across Israel as Iran launches drones and missiles TEL AVIV, Israel - Israel has suffered casualties as Iran fights back from the Jewish state's unprecedented unilateral attacks which... Scores of Israeli planes swarm over Iran in surprise invasion WEST JERUSALEM, Israel - The Israel Air Force has launched a pre-emptive Pearl Harbour style air raid over Iran, dropping bombs over... Hundreds of lives lost as London-bound Air India plane crashes after take-off NEW DELHI, India - The world is in shock following Thursday's devastating plane crash in India which has killed at least 290 people,... Business Section Chime soars in debut, giving fintech sector a much-needed boost SAN FRANCISCO, California: After months of muted IPO activity in the fintech world, digital bank Chime Financial reignited investor... Switch 2 breaks Nintendo record with 3.5M units sold in 4 days TOKYO, Japan: Nintendo's latest console is off to a roaring start. The company says it has sold over 3.5 million units of the newly... U.S.-China trade progress lifts oil prices to multi-week peak LONDON, UK: Crude prices surged this week as investors welcomed fresh signs of progress in U.S.-China relations, lifting hopes of reduced... Debt fears drive fund outflows from US, inflows to Europe NEW YORK CITY, New York: Investor confidence in U.S. markets is showing signs of strain as global funds redirect billions toward Europe... Japan celebrates new sumo grand champion TOKYO, Japan: Japan has a new top sumo wrestler — and he's Japanese. Onosato, who weighs 191 kilograms (421 pounds), has become a yokozuna,... TikTok star Khaby Lame detained in Las Vegas for overstaying visa LAS VEGAS, Nevada: Khaby Lame, the most followed person on TikTok with millions of fans, has left the United States after being held... Movie Review Deep Water"

    ]

    analisis = analizar_sentimiento(noticias)

    for res in analisis:
        print(f"\nNoticia: {res['texto']}")
        print(f"Sentimiento: {res['sentiment'].upper()} (Pos: {res['score_positive']:.2f}, Neg: {res['score_negative']:.2f})")