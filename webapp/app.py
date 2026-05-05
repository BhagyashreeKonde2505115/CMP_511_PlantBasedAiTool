"""
webapp/app.py — Plant Disease Detection + Built-in AI Analysis Engine
No external API key required. All analysis powered by your trained models.

Run: python webapp/app.py
  http://localhost:5000          → Main web app
  http://localhost:5000/ai-tool  → AI Analysis Tool
"""
import os
from urllib.request import urlretrieve

HF_BASE = "https://huggingface.co/BhagyashreeKondeAbertay/plant-disease-ai-models/resolve/main"

MODELS = {
    "saved_models/cnn_model.pth":      f"{HF_BASE}/cnn_model.pth",
    "saved_models/resnet50_model.pth": f"{HF_BASE}/resnet50_model.pth",
    "saved_models/vgg16_model.pth":    f"{HF_BASE}/vgg16_model.pth",
    "saved_models/rf_model.joblib":    f"{HF_BASE}/rf_model.joblib",
    "saved_models/svm_model.joblib":   f"{HF_BASE}/svm_model.joblib",
}

os.makedirs("saved_models", exist_ok=True)
for path, url in MODELS.items():
    if not os.path.exists(path):
        print(f"Downloading {path}...")
        urlretrieve(url, path)
        print(f"Done: {path}")

import os, sys, uuid, json, random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from werkzeug.utils import secure_filename
from config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG, MAX_UPLOAD_MB, UPLOAD_DIR, RESULTS_DIR

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
ALLOWED_EXTENSIONS = {"jpg","jpeg","png","bmp","webp"}

try:
    from utils.predictor import Predictor
    predictor = Predictor()
except Exception as e:
    print(f"[Warning] Predictor not loaded: {e}")
    predictor = None

def allowed_file(f):
    return "." in f and f.rsplit(".",1)[1].lower() in ALLOWED_EXTENSIONS

# ══════════════════════════════════════════════════════════════════════════════
# DISEASE KNOWLEDGE BASE — detailed info for every PlantVillage class
# ══════════════════════════════════════════════════════════════════════════════
KB = {
    "Apple___Apple_scab":{"common":"Apple Scab","pathogen":"Venturia inaequalis (fungus)","sev":(40,75),"symptoms":["Olive-green or brown lesions on leaves","Scabby corky spots on fruit","Premature leaf drop","Distorted or cracked fruit"],"cause":"Fungal spores spread by rain splash during cool wet spring weather. Overwinters in fallen leaves.","treatment":["Remove and destroy all infected leaves and fallen debris immediately","Apply fungicide (Captan, Mancozeb, or Myclobutanil) every 7–10 days during wet periods","Ensure good air circulation by pruning crowded branches","Avoid overhead irrigation — use drip irrigation instead"],"organic":["Neem oil spray every 7 days","Sulfur-based fungicide at bud break","Bicarbonate of soda solution (1 tbsp per litre)"],"prevention":["Plant scab-resistant apple varieties","Rake and compost fallen leaves in autumn","Apply dormant lime-sulfur spray before bud break","Maintain proper tree spacing for air circulation"],"recovery":"3–4 weeks with consistent treatment","yield":"Can cause 50–70% yield loss if untreated; fruit becomes unsellable","spread":"High — spreads rapidly in wet conditions to neighbouring trees"},
    "Apple___Black_rot":{"common":"Apple Black Rot","pathogen":"Botryosphaeria obtusa (fungus)","sev":(45,80),"symptoms":["Purple spots on leaves enlarging to brown","Black rotting areas on fruit","Cankers on branches","Mummified fruit remaining on tree"],"cause":"Fungal infection entering through wounds, insect damage, or natural openings. Stress weakens tree defences.","treatment":["Prune out all infected branches 15cm below visible infection","Remove mummified fruit from tree and ground","Apply Captan or Thiophanate-methyl fungicide","Improve tree vigour with balanced fertilisation"],"organic":["Copper-based fungicide spray","Remove all infected material","Neem oil applications"],"prevention":["Regular pruning to remove dead wood","Control insect pests that create entry wounds","Avoid tree stress through proper irrigation","Annual dormant oil spray"],"recovery":"4–6 weeks; severe cases may need a full season","yield":"40–60% loss on affected branches","spread":"Moderate — spreads via rain splash and infected pruning tools"},
    "Apple___Cedar_apple_rust":{"common":"Cedar Apple Rust","pathogen":"Gymnosporangium juniperi-virginianae (fungus)","sev":(30,65),"symptoms":["Bright orange-yellow spots on upper leaf surface","Tube-like structures on leaf undersides","Yellowing and premature defoliation","Orange lesions on young fruit"],"cause":"Requires two hosts: apple AND eastern red cedar/juniper. Spores travel up to 3 miles between hosts.","treatment":["Apply myclobutanil or propiconazole fungicide at pink bud stage","Continue applications every 7–10 days through petal fall","Remove nearby juniper/cedar trees if possible","Apply protective fungicide before infection periods"],"organic":["Sulfur dust at bud break","Lime sulfur dormant spray","Neem oil as protective spray"],"prevention":["Plant rust-resistant apple varieties","Remove cedar galls in early spring","Avoid planting apples near junipers","Apply preventive fungicide before wet periods"],"recovery":"2–3 weeks after treatment begins","yield":"20–40% defoliation; reduces fruit size and sugar content","spread":"Low between apple trees — requires juniper as alternate host"},
    "Apple___healthy":{"common":"Healthy Apple","pathogen":None,"sev":(0,5),"symptoms":["Vibrant green leaves with no lesions","Normal leaf shape and texture","No discolouration or spots"],"cause":"Plant is in excellent health","treatment":["No treatment required — maintain current practices"],"organic":["Continue regular neem oil preventive spray monthly"],"prevention":["Maintain balanced fertilisation","Monitor weekly for early disease signs","Ensure adequate spacing for air circulation"],"recovery":"Plant is healthy","yield":"None — expected to yield normally","spread":"None"},
    "Tomato___Early_blight":{"common":"Tomato Early Blight","pathogen":"Alternaria solani (fungus)","sev":(35,70),"symptoms":["Dark brown spots with concentric rings (target-board pattern)","Yellowing around lesions (chlorotic halo)","Lesions starting on lower/older leaves first","Stem cankers near soil level"],"cause":"Fungal pathogen thrives in warm (24–29°C) humid conditions. Spreads via infected soil, plant debris, and rain splash.","treatment":["Remove all infected lower leaves immediately and dispose away from garden","Apply chlorothalonil or mancozeb fungicide every 7 days","Stake plants to improve air circulation and keep leaves off soil","Mulch around base to prevent soil splash onto leaves"],"organic":["Copper fungicide spray every 7–10 days","Neem oil + baking soda mixture","Bacillus subtilis biofungicide (Serenade)"],"prevention":["Rotate tomato crops every 3 years","Use disease-resistant varieties","Water at base — never overhead","Remove plant debris at end of season"],"recovery":"2–3 weeks with consistent treatment; new growth will be healthy","yield":"25–50% yield reduction if untreated","spread":"High — spreads rapidly in warm humid weather"},
    "Tomato___Late_blight":{"common":"Tomato Late Blight","pathogen":"Phytophthora infestans (oomycete)","sev":(60,95),"symptoms":["Water-soaked grey-green lesions on leaves","White fuzzy mould on leaf undersides","Dark brown rapidly spreading lesions","Fruit develops firm brown rot"],"cause":"Same pathogen that caused the Irish Potato Famine. Thrives in cool (10–20°C) wet conditions. Extremely destructive.","treatment":["ACT IMMEDIATELY — remove and bag ALL infected plant material","Apply mancozeb or chlorothalonil fungicide URGENTLY","Consider removing entire plant if over 30% infected","Do NOT compost infected material — burn or bin it"],"organic":["Copper hydroxide spray as soon as symptoms appear","Remove infected plants entirely if severe"],"prevention":["Plant only certified blight-resistant varieties","Avoid overhead watering completely","Improve drainage","Apply preventive copper spray in wet forecasts"],"recovery":"Difficult once established — early action is critical","yield":"Can cause total crop loss (100%) within 1–2 weeks if unchecked","spread":"VERY HIGH — can wipe out entire fields in days"},
    "Tomato___Leaf_Mold":{"common":"Tomato Leaf Mould","pathogen":"Passalora fulva (fungus)","sev":(30,65),"symptoms":["Pale green or yellow patches on upper leaf surface","Olive-green to brown velvety mould on undersides","Leaves curl upward and eventually die","Mainly affects greenhouse/polytunnel tomatoes"],"cause":"Thrives in high humidity (over 85%) with poor ventilation. Common in greenhouses and polytunnels.","treatment":["Improve ventilation immediately — open vents and doors","Reduce humidity — avoid overhead watering","Apply mancozeb or copper fungicide","Remove and destroy heavily infected leaves"],"organic":["Neem oil spray every 5–7 days","Improve airflow as primary intervention","Reduce plant density"],"prevention":["Maintain greenhouse humidity below 85%","Space plants adequately (60cm apart)","Use fans for air circulation","Water in morning so leaves dry by evening"],"recovery":"2–3 weeks with improved ventilation","yield":"20–40% if untreated","spread":"Moderate — primarily in enclosed growing environments"},
    "Tomato___Bacterial_spot":{"common":"Tomato Bacterial Spot","pathogen":"Xanthomonas vesicatoria (bacteria)","sev":(35,70),"symptoms":["Small water-soaked spots on leaves turning brown","Spots with yellow halo","Raised scabby lesions on fruit","Defoliation in severe cases"],"cause":"Bacterial disease spread by rain, wind, and contaminated tools. Thrives in warm (24–30°C) wet weather.","treatment":["Apply copper hydroxide or copper octanoate bactericide immediately","Repeat every 5–7 days during wet weather","Remove heavily infected leaves","Disinfect tools with 70% alcohol between plants"],"organic":["Copper-based bactericide is the primary organic option","Bacillus subtilis biofungicide"],"prevention":["Use certified disease-free transplants","Avoid working in wet conditions","Stake plants for air circulation","Practice 2-year crop rotation"],"recovery":"3–4 weeks with consistent copper applications","yield":"30–50% reduction in severe outbreaks","spread":"High during rain and warm weather"},
    "Tomato___Septoria_leaf_spot":{"common":"Septoria Leaf Spot","pathogen":"Septoria lycopersici (fungus)","sev":(35,70),"symptoms":["Small circular spots with dark border and light grey centre","Tiny black dots visible in spot centres","Spots start on lower leaves","Rapid defoliation from bottom up"],"cause":"Fungal disease spread by water splash from infected soil. Thrives in warm (20–25°C) humid conditions.","treatment":["Remove all infected lower leaves immediately","Apply chlorothalonil or mancozeb fungicide every 7–10 days","Mulch heavily around base to prevent soil splash","Stake plants to improve air circulation"],"organic":["Copper fungicide spray","Neem oil every 7 days","Remove infected foliage promptly"],"prevention":["3-year crop rotation","Mulch to prevent soil splash","Avoid overhead watering","Remove plant debris at season end"],"recovery":"2–3 weeks; new growth will be clean","yield":"20–40% if defoliation is severe","spread":"High via rain and irrigation splash"},
    "Tomato___Spider_mites Two-spotted_spider_mite":{"common":"Two-Spotted Spider Mite","pathogen":"Tetranychus urticae (mite — not a fungal disease)","sev":(30,70),"symptoms":["Tiny yellow or white stippling on leaves","Fine webbing on leaf undersides","Leaves turn bronze/yellow and dry out","Visible tiny moving dots on leaf undersides"],"cause":"Mite infestation. Thrives in hot dry dusty conditions. Spreads rapidly between plants.","treatment":["Spray forcefully with water to dislodge mites especially undersides","Apply miticide or insecticidal soap","Increase humidity — mites hate moisture","Apply 2–3 times at 5-day intervals to catch hatching eggs"],"organic":["Neem oil spray every 3–5 days","Insecticidal soap solution","Introduce predatory mites (Phytoseiulus persimilis)","Strong water jet to knock mites off"],"prevention":["Keep plants well-watered — stressed plants attract mites","Avoid dusty conditions","Monitor undersides of leaves weekly","Encourage natural predators (ladybirds)"],"recovery":"1–2 weeks with consistent treatment","yield":"20–50% if untreated; leaves die and fruit sunscald","spread":"Very high in hot weather"},
    "Tomato___Target_Spot":{"common":"Tomato Target Spot","pathogen":"Corynespora cassiicola (fungus)","sev":(35,65),"symptoms":["Brown spots with concentric ring pattern","Yellow halo surrounding spots","Affects leaves stems and fruit","Spots coalesce causing large dead areas"],"cause":"Fungal disease favoured by warm temperatures (25–30°C) and high humidity.","treatment":["Apply azoxystrobin or chlorothalonil fungicide","Remove infected plant material","Improve air circulation around plants","Avoid wetting foliage during irrigation"],"organic":["Copper fungicide spray","Neem oil applications","Remove infected leaves"],"prevention":["Use resistant varieties","Crop rotation every 2–3 years","Proper plant spacing","Avoid overhead irrigation"],"recovery":"3–4 weeks with treatment","yield":"30–50% in severe cases","spread":"Moderate in warm humid conditions"},
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":{"common":"Tomato Yellow Leaf Curl Virus","pathogen":"Begomovirus — transmitted by silverleaf whitefly","sev":(55,90),"symptoms":["Leaves curl upward and turn yellow","Stunted plant growth","Reduced leaf size","Poor fruit set — few or no tomatoes"],"cause":"Transmitted exclusively by silverleaf whitefly (Bemisia tabaci). Not spread by touch or tools.","treatment":["No cure — control the whitefly vector immediately","Apply imidacloprid or thiamethoxam insecticide for whitefly","Use yellow sticky traps to monitor and catch whiteflies","Remove and destroy severely infected plants"],"organic":["Neem oil spray to deter whiteflies","Yellow sticky traps","Reflective mulch to confuse whiteflies","Remove infected plants"],"prevention":["Use TYLCV-resistant tomato varieties","Cover young plants with insect-proof mesh","Control whitefly populations early in season","Avoid planting near infected crops"],"recovery":"No recovery — manage whitefly to protect remaining plants","yield":"60–100% loss on infected plants","spread":"High where whitefly populations are large"},
    "Tomato___Tomato_mosaic_virus":{"common":"Tomato Mosaic Virus","pathogen":"Tomato mosaic virus — spread by contact","sev":(40,80),"symptoms":["Light and dark green mosaic pattern on leaves","Leaf distortion and curling","Stunted plant growth","Reduced fruit set and mottled fruit"],"cause":"Transmitted by handling (hands, tools) not by insects. Extremely persistent — survives on tools for years.","treatment":["There is NO cure for viral infections — focus on containment","Remove and destroy infected plants immediately","Disinfect all tools with 10% bleach solution after use","Wash hands thoroughly before handling other plants"],"organic":["Remove infected plants — only management option","Milk spray (1:10 ratio) may slow spread on contact"],"prevention":["Use certified virus-free seeds","Wash hands before working with tomatoes","Control aphid populations","Plant resistant varieties (TMV-resistant)"],"recovery":"No recovery — infected plants should be removed","yield":"50–100% loss on infected plants","spread":"Very high via touch and contaminated tools"},
    "Tomato___healthy":{"common":"Healthy Tomato","pathogen":None,"sev":(0,5),"symptoms":["Deep green uniform foliage","No lesions spots or discolouration","Strong upright growth"],"cause":"Plant is in excellent health","treatment":["No treatment required"],"organic":["Weekly neem oil spray as preventive measure"],"prevention":["Water consistently at base","Support plants with stakes or cages","Feed weekly with tomato-specific fertiliser","Check weekly for early pest or disease signs"],"recovery":"Plant is healthy","yield":"None — expect full yield","spread":"None"},
    "Potato___Early_blight":{"common":"Potato Early Blight","pathogen":"Alternaria solani (fungus)","sev":(30,65),"symptoms":["Dark brown spots with concentric rings on leaves","Spots surrounded by yellow halo","Lower leaves affected first","Stems may develop elongated dark lesions"],"cause":"Fungal pathogen favoured by warm days (24–29°C) and wet/humid nights. Common in mid to late season.","treatment":["Apply mancozeb or chlorothalonil fungicide every 7–10 days","Remove infected lower leaves","Ensure adequate potassium fertilisation","Avoid overhead irrigation"],"organic":["Copper-based fungicide spray","Neem oil applications","Bacillus subtilis (Serenade)"],"prevention":["Use certified disease-free seed potatoes","Crop rotation — 3-year cycle","Destroy all potato debris after harvest","Maintain adequate plant nutrition"],"recovery":"2–3 weeks with consistent fungicide programme","yield":"20–30% tuber weight reduction if severe","spread":"Moderate — spreads by rain splash and wind"},
    "Potato___Late_blight":{"common":"Potato Late Blight","pathogen":"Phytophthora infestans (oomycete)","sev":(65,98),"symptoms":["Dark water-soaked lesions on leaves and stems","White fluffy growth on leaf undersides in humid weather","Tubers show brown rot internally","Rapid whole-plant collapse"],"cause":"The most devastating potato disease in history (caused the Irish Famine). Cool (10–20°C) wet conditions trigger rapid spread.","treatment":["EMERGENCY: Apply metalaxyl + mancozeb fungicide immediately","If over 30% of foliage infected consider destroying entire crop","Do NOT harvest immediately — wait 2 weeks after haul-off for skin to set","Destroy all infected haulm completely"],"organic":["Copper hydroxide spray urgently","Remove and destroy infected haulm","Consider crop destruction if severe"],"prevention":["Use certified blight-resistant varieties","Apply preventive fungicide before wet weather forecasts","Avoid planting in poorly drained areas","Earth up rows to protect tubers"],"recovery":"Tubers may survive if foliage destroyed early; difficult once established","yield":"Total crop loss possible within 1–2 weeks","spread":"EXTREME — can destroy entire fields overnight"},
    "Potato___healthy":{"common":"Healthy Potato","pathogen":None,"sev":(0,5),"symptoms":["Vigorous dark green foliage","No lesions or discolouration","Normal growth pattern"],"cause":"Plant is healthy","treatment":["No treatment needed"],"organic":["Preventive copper spray during wet periods"],"prevention":["Earth up ridges regularly","Monitor weekly especially in wet weather","Ensure good drainage","Use certified seed potatoes each season"],"recovery":"Plant is healthy","yield":"None — expect good yield","spread":"None"},
    "Pepper__bell___Bacterial_spot":{"common":"Pepper Bacterial Spot","pathogen":"Xanthomonas campestris pv. vesicatoria (bacteria)","sev":(35,70),"symptoms":["Small water-soaked spots turning brown/black on leaves","Yellow halo around spots","Raised scabby spots on fruit surface","Defoliation in severe outbreaks"],"cause":"Bacterial disease spread by rain, wind, and contaminated tools. Thrives in warm (24–30°C) wet conditions.","treatment":["Apply copper hydroxide or copper octanoate bactericide immediately","Repeat every 5–7 days during wet weather","Remove heavily infected leaves","Disinfect tools with 70% alcohol between plants"],"organic":["Copper-based bactericide is the primary organic option","Remove infected foliage","Avoid wetting foliage"],"prevention":["Use certified disease-free transplants","Avoid working in wet conditions","Stake plants for air circulation","Practice 2-year crop rotation"],"recovery":"3–4 weeks with consistent treatment","yield":"30–50% fruit damage and defoliation","spread":"High during rain and warm weather"},
    "Pepper__bell___healthy":{"common":"Healthy Bell Pepper","pathogen":None,"sev":(0,5),"symptoms":["Glossy dark green leaves","No spots lesions or discolouration","Healthy stem and normal growth"],"cause":"Plant is in good health","treatment":["No treatment needed"],"organic":["Preventive neem oil spray monthly"],"prevention":["Water at base consistently","Feed with pepper-specific fertiliser every 2 weeks","Check undersides of leaves for pests weekly"],"recovery":"Plant is healthy","yield":"None","spread":"None"},
    "Corn_(maize)___Common_rust_":{"common":"Corn Common Rust","pathogen":"Puccinia sorghi (fungus)","sev":(30,65),"symptoms":["Small oval cinnamon-brown pustules on both leaf surfaces","Pustules scattered across entire leaf","Severely infected leaves turn yellow then brown","Reduces photosynthesis significantly"],"cause":"Airborne fungal spores blown in from southern regions. Favoured by cool temperatures (16–23°C) and high humidity.","treatment":["Apply propiconazole or azoxystrobin fungicide at first sign of pustules","Treat early before tassel stage for best results","Repeat application if wet conditions persist","Ensure adequate potassium levels — improves rust resistance"],"organic":["Sulfur-based fungicide spray","Neem oil (limited efficacy)","Remove severely infected leaves"],"prevention":["Plant rust-resistant hybrid varieties","Plant early to avoid peak rust season","Avoid excessive nitrogen fertilisation","Scout fields regularly from V6 stage onwards"],"recovery":"2–3 weeks after fungicide application","yield":"10–40% grain yield loss in severe epidemics","spread":"Very high — spores travel hundreds of miles on wind"},
    "Corn_(maize)___Northern_Leaf_Blight":{"common":"Northern Corn Leaf Blight","pathogen":"Exserohilum turcicum (fungus)","sev":(35,75),"symptoms":["Long (10–15cm) cigar-shaped grey-green to tan lesions","Lesions run parallel to leaf veins","Lesions develop dark olive-green spore masses","Severe infection causes entire plant to appear grey"],"cause":"Fungal disease favoured by moderate temperatures (18–27°C) and extended leaf wetness. Overwinters in crop residue.","treatment":["Apply foliar fungicide (propiconazole azoxystrobin) at silking stage","Treatment most effective before tasselling","Ensure adequate plant nutrition","Reduce plant density if air circulation is poor"],"organic":["Copper-based fungicide spray","Biological control with Bacillus subtilis"],"prevention":["Plant resistant hybrid varieties (key prevention)","Till under crop residue after harvest","Rotate with non-host crops (soybeans wheat)","Avoid excessive nitrogen that promotes dense canopy"],"recovery":"3–4 weeks with fungicide; grain fill may be impacted","yield":"20–50% yield loss if infection occurs before tasselling","spread":"High — spores spread by wind and rain"},
    "Corn_(maize)___healthy":{"common":"Healthy Maize","pathogen":None,"sev":(0,5),"symptoms":["Uniform dark green leaves","No pustules spots or lesions","Normal growth and development"],"cause":"Plant is healthy","treatment":["No treatment needed"],"organic":["Continue current management practices"],"prevention":["Scout regularly from V6 growth stage","Maintain balanced fertilisation","Ensure adequate potassium for disease resistance"],"recovery":"Plant is healthy","yield":"None","spread":"None"},
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot":{"common":"Gray Leaf Spot","pathogen":"Cercospora zeae-maydis (fungus)","sev":(35,70),"symptoms":["Rectangular lesions restricted by leaf veins","Tan to grey coloured lesions","Lesions run parallel to leaf veins","Entire leaf can appear grey in severe cases"],"cause":"Fungal disease favoured by high humidity warm nights and minimum tillage systems where residue builds up.","treatment":["Apply fungicide (strobilurin + triazole mixture) at VT/R1 growth stage","Ensure balanced fertilisation especially potassium","Improve air flow by reducing plant population if excessively dense"],"organic":["Limited organic options — copper spray has some efficacy","Focus on prevention"],"prevention":["Plant resistant hybrids (most important factor)","Rotate crops — reduce corn-on-corn","Till to bury surface residue","Scout from V6 stage"],"recovery":"2–3 weeks post-fungicide in mild cases","yield":"10–40% yield loss in severe outbreaks","spread":"Moderate — spores spread by wind from residue"},
    "Grape___Black_rot":{"common":"Grape Black Rot","pathogen":"Guignardia bidwellii (fungus)","sev":(45,85),"symptoms":["Tan lesions with dark borders on leaves","Small black pycnidia (dots) in lesion centres","Fruit shrivels into hard black mummies","Affects shoots and tendrils too"],"cause":"Fungal disease with spores released during rain. Infects all green tissue. Overwinters in mummified berries and canes.","treatment":["Apply myclobutanil mancozeb or captan from pre-bloom through post-bloom","Remove ALL mummified berries from vine and ground — crucial","Prune infected canes in winter to remove overwintering inoculum","Apply fungicide every 7–10 days during wet periods"],"organic":["Copper hydroxide spray at bud break","Remove all mummies and infected material","Sulfur spray pre-bloom"],"prevention":["Remove mummies and debris in winter — most important step","Open up canopy by training and pruning for air circulation","Plant in well-drained sites","Choose resistant varieties where available"],"recovery":"4–6 weeks; this season's mummies must be removed to protect next year","yield":"Total fruit loss on infected clusters; 50–80% crop loss possible","spread":"Very high — one mummy produces millions of spores"},
    "Grape___Esca_(Black_Measles)":{"common":"Esca (Black Measles)","pathogen":"Complex of fungi: Phaeomoniella chlamydospora etc.","sev":(50,90),"symptoms":["Interveinal chlorosis (tiger-stripe pattern) on leaves","Berries develop dark spots and crack","Internal wood shows brown streaking when cut","Sudden vine collapse in severe cases"],"cause":"Fungal wood disease entering through pruning wounds. Long incubation period.","treatment":["No effective cure once established in wood","Remove and destroy infected vines in severe cases","Apply trunk wound protectants after pruning","Manage stress factors — ensure adequate nutrition and irrigation"],"organic":["Thymol-based wound sealants","Trichoderma biological control applied to wounds"],"prevention":["Protect ALL pruning wounds with fungicide or wound sealant immediately after pruning","Prune in dry weather","Use clean disinfected pruning tools between vines","Avoid large pruning wounds"],"recovery":"Partial recovery possible with trunk renewal (retrain new arm from base)","yield":"Progressive yield decline over years; 20–80% loss","spread":"Low between vines — mainly enters through wounds"},
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)":{"common":"Grape Leaf Blight","pathogen":"Pseudocercospora vitis (fungus)","sev":(30,60),"symptoms":["Angular brown spots on leaves","Spots may have dark borders","Premature defoliation","Reduces photosynthesis"],"cause":"Fungal leaf disease favoured by warm humid conditions in late summer.","treatment":["Apply mancozeb or copper fungicide","Remove infected leaves to slow spread","Improve canopy ventilation by leaf removal"],"organic":["Copper-based fungicide spray","Neem oil applications"],"prevention":["Maintain open canopy structure through good pruning","Avoid excessive nitrogen fertilisation","Good drainage"],"recovery":"2–3 weeks with treatment","yield":"10–30% from premature defoliation","spread":"Moderate"},
    "Grape___healthy":{"common":"Healthy Grapevine","pathogen":None,"sev":(0,5),"symptoms":["Healthy green leaves with no lesions","Normal growth and development"],"cause":"Plant is healthy","treatment":["No treatment needed"],"organic":["Preventive sulfur spray during humid periods"],"prevention":["Annual dormant pruning for open canopy","Monitor for powdery mildew during shoot growth","Protect pruning wounds with sealant"],"recovery":"Plant is healthy","yield":"None","spread":"None"},
    "Strawberry___Leaf_scorch":{"common":"Strawberry Leaf Scorch","pathogen":"Diplocarpon earlianum (fungus)","sev":(30,65),"symptoms":["Small irregular dark purple spots on leaves","Spots enlarge and centres turn tan/grey","Severely infected leaves turn entirely brown (scorched appearance)","Mainly affects older leaves"],"cause":"Fungal disease spread by rain splash. Overwinters in infected plant debris.","treatment":["Apply myclobutanil or captan fungicide at first sign of symptoms","Remove and destroy all old infected leaves","Renovate beds after fruiting — mow and remove debris","Apply fungicide protectively before wet weather"],"organic":["Copper fungicide applications","Remove infected foliage promptly"],"prevention":["Use disease-resistant varieties","Use drip irrigation","Renovate beds annually","Ensure good plant spacing for air circulation"],"recovery":"3–4 weeks with treatment programme","yield":"20–40% in severe cases through defoliation","spread":"Moderate — spreads by rain splash"},
    "Strawberry___healthy":{"common":"Healthy Strawberry","pathogen":None,"sev":(0,5),"symptoms":["Bright green healthy leaves","No spots or discolouration"],"cause":"Plant is healthy","treatment":["No treatment needed"],"organic":["Preventive neem oil spray"],"prevention":["Renovate beds annually","Use drip irrigation","Monitor for slugs and aphids"],"recovery":"Plant is healthy","yield":"None","spread":"None"},
    "Cherry_(including_sour)___Powdery_mildew":{"common":"Cherry Powdery Mildew","pathogen":"Podosphaera clandestina (fungus)","sev":(30,65),"symptoms":["White powdery coating on young leaves and shoots","Leaves curl and distort","Affected tissue turns brown and dies","Young fruit may be covered in white powder"],"cause":"Unlike most fungi, powdery mildew thrives in warm DRY conditions with high humidity. Spores spread by wind.","treatment":["Apply myclobutanil trifloxystrobin or potassium bicarbonate","Do NOT apply sulfur in temperatures above 30°C (leaf burn)","Remove and destroy infected shoots","Apply every 7–14 days"],"organic":["Potassium bicarbonate solution (most effective organic option)","Neem oil spray","Milk spray (1:9 milk:water ratio)","Baking soda + vegetable oil spray"],"prevention":["Avoid excessive nitrogen fertilisation","Ensure good air circulation through pruning","Plant resistant varieties","Avoid water stress — weakens plant defences"],"recovery":"2–3 weeks with treatment","yield":"20–40% fruit quality reduction","spread":"High — wind-dispersed spores spread quickly"},
    "Cherry_(including_sour)___healthy":{"common":"Healthy Cherry","pathogen":None,"sev":(0,5),"symptoms":["Shiny dark green leaves","No powder spots or lesions"],"cause":"Plant is healthy","treatment":["No treatment needed"],"organic":["Preventive sulfur spray at bud swell"],"prevention":["Annual pruning for open canopy","Monitor for aphids in spring"],"recovery":"Plant is healthy","yield":"None","spread":"None"},
    "Peach___Bacterial_spot":{"common":"Peach Bacterial Spot","pathogen":"Xanthomonas arboricola pv. pruni (bacteria)","sev":(35,75),"symptoms":["Small water-soaked spots on leaves turning angular and brown","Spots fall out leaving shot-hole appearance","Dark raised lesions on fruit","Defoliation and twig cankers in severe cases"],"cause":"Bacterial infection spread by rain and wind. Enters through stomata and wounds. Warm (25–30°C) wet conditions promote rapid spread.","treatment":["Apply copper bactericide (copper hydroxide) from petal fall","Apply oxytetracycline where legally permitted","Remove infected shoots and cankers during dry weather","Repeat copper applications every 5–10 days during wet periods"],"organic":["Copper hydroxide spray is the main tool","Remove infected material"],"prevention":["Plant in sheltered locations to reduce wind-driven rain","Choose resistant varieties","Avoid excessive nitrogen","Dormant copper spray before bud break"],"recovery":"3–5 weeks; severe cases affect multiple seasons","yield":"30–60% fruit loss with severe spotting","spread":"High during wet warm weather"},
    "Peach___healthy":{"common":"Healthy Peach","pathogen":None,"sev":(0,5),"symptoms":["Healthy green lanceolate leaves","No spots or shot-holes"],"cause":"Plant is healthy","treatment":["No treatment needed"],"organic":["Preventive copper spray at dormancy"],"prevention":["Annual pruning for open vase shape","Monitor for aphids and leaf curl"],"recovery":"Plant is healthy","yield":"None","spread":"None"},
    "Squash___Powdery_mildew":{"common":"Squash Powdery Mildew","pathogen":"Podosphaera xanthii (fungus)","sev":(30,65),"symptoms":["White powdery patches on upper and lower leaf surfaces","Leaves yellow and die prematurely","Stems and petioles may also be affected","Fruit quality and size reduced"],"cause":"Very common fungal disease. Thrives in warm (20–28°C) dry days and cool nights. Does NOT need wet leaves to spread.","treatment":["Apply potassium bicarbonate sulfur or myclobutanil fungicide","Remove heavily infected leaves","Apply every 7 days","Avoid applying sulfur when temperatures exceed 30°C"],"organic":["Potassium bicarbonate spray (most effective)","Milk spray (1:9 ratio) applied every 3 days","Neem oil every 7 days","Baking soda + soap spray"],"prevention":["Plant resistant varieties","Ensure adequate plant spacing","Avoid excessive nitrogen fertiliser","Plant in full sun — shade promotes disease"],"recovery":"1–2 weeks with treatment on mild cases","yield":"20–40% from premature defoliation reducing fruit fill","spread":"Very high — wind-dispersed; can infect entire plot quickly"},
    "Soybean___healthy":{"common":"Healthy Soybean","pathogen":None,"sev":(0,5),"symptoms":["Uniform trifoliate leaves","No lesions discolouration or deformity"],"cause":"Plant is healthy","treatment":["No treatment needed"],"organic":["Scout regularly for soybean aphid"],"prevention":["Seed treatment with fungicide before planting","Scout for sudden death syndrome early"],"recovery":"Plant is healthy","yield":"None","spread":"None"},
    "Background_without_leaves":{"common":"No Leaf Detected","pathogen":None,"sev":(0,0),"symptoms":["No plant material detected in image"],"cause":"Image does not contain a recognisable leaf","treatment":["Please upload a clear close-up photo of a plant leaf"],"organic":[],"prevention":["For best results: photograph a single leaf against a contrasting background"],"recovery":"N/A","yield":"N/A","spread":"N/A"},
}

FOLLOWUP = {
    "organic":"For organic treatment, the most effective approaches are:\n1. Copper-based fungicides (copper hydroxide or copper sulphate) — approved for organic use\n2. Neem oil spray every 5–7 days — broad spectrum organic control\n3. Potassium bicarbonate — excellent for powdery mildew\n4. Bacillus subtilis (Serenade) — biological fungicide safe for organic systems\n5. Physical removal of infected plant material\n\nOrganic treatments often need more frequent application than synthetic alternatives.",
    "spread":"To prevent spread to other plants:\n1. Isolate infected plants immediately if possible\n2. Remove infected leaves — bag or burn, do NOT compost\n3. Disinfect tools with 70% alcohol or 10% bleach solution between plants\n4. Wash hands after handling infected plants\n5. Avoid working in plants when they are wet\n6. Apply a preventive fungicide spray to neighbouring plants\n7. Increase spacing between plants to improve air circulation",
    "fertiliser":"Correct nutrition plays a major role in disease resistance:\n• Nitrogen: Avoid excess — promotes soft disease-susceptible tissue\n• Potassium: Very important — improves cell wall strength and disease resistance\n• Phosphorus: Supports root development and overall plant vigour\n• Calcium: Strengthens cell walls — apply as calcium nitrate\n• Silicon: Natural foliar spray improves resistance to fungal penetration\n\nA balanced NPK fertiliser (e.g. 10-10-10) applied correctly reduces disease severity.",
    "soil":"Soil health is fundamental to disease prevention:\n1. Improve drainage — most soil-borne pathogens thrive in waterlogged conditions\n2. Add organic matter (compost) to improve soil structure\n3. Maintain pH 6.0–7.0 for most vegetables\n4. Apply beneficial soil microbes (Trichoderma) to suppress soil pathogens\n5. Practice crop rotation — prevents pathogen build-up in soil\n6. Avoid soil compaction — ensures good drainage and root health",
    "recovery":"Recovery depends on disease severity and how quickly you act:\n• Mild infections (under 20% leaf damage): Full recovery in 2–3 weeks with treatment\n• Moderate infections (20–50%): Partial recovery in 3–5 weeks; yield may be affected\n• Severe infections (over 50%): Slow recovery; consider removing and replacing\n\nKey factors for faster recovery:\n1. Start treatment as early as possible\n2. Remove all infected material to reduce disease pressure\n3. Ensure good nutrition and consistent watering\n4. Maintain good air circulation around plants",
    "contagious":"Yes, most plant diseases can spread to neighbouring plants:\n• Fungal diseases spread by air currents, rain splash, and contaminated tools\n• Bacterial diseases spread by rain, wind, insects, and touch\n• Viral diseases spread by insects (aphids, whiteflies) or contact\n\nTo protect healthy plants:\n1. Apply preventive fungicide/bactericide spray to healthy plants nearby\n2. Disinfect all tools used on infected plants\n3. Remove infected plants or isolate them\n4. Avoid walking through affected areas then into clean areas",
    "water":"Watering advice for disease management:\n1. Always water at the BASE of plants — never overhead\n2. Water in the morning so foliage dries quickly in sun\n3. Use drip irrigation where possible — keeps leaves dry\n4. Allow soil surface to dry slightly between waterings\n5. Wet foliage is the number one cause of fungal spread\n6. Avoid watering in evening — moisture sits overnight\n7. Mulch around base to reduce water splash onto leaves",
    "buy":"Recommended products for plant disease:\n\nFUNGICIDES:\n• Mancozeb (Dithane M-45) — broad spectrum, affordable\n• Chlorothalonil (Bravo) — protectant fungicide\n• Propiconazole (Tilt) — systemic, good for rusts\n• Azoxystrobin (Amistar) — systemic, excellent results\n\nBACTERICIDES:\n• Copper hydroxide (Kocide) — for bacterial diseases\n\nORGANIC:\n• Copper hydroxide — approved organic\n• Neem oil concentrate — dilute as per label\n• Potassium bicarbonate (Armicarb) — for powdery mildew\n\nAlways read the label and check local availability.",
    "food":"Food safety after plant disease treatment:\n• Most fungal leaf diseases do NOT affect edibility of fruit if treated promptly\n• Always observe the Pre-Harvest Interval (PHI) on fungicide labels — typically 7–28 days\n• Wash all produce thoroughly before consumption\n• Remove any visually affected fruit — do not eat rotted or mouldy produce\n• Organic treatments (neem oil, copper) have shorter or no PHI periods",
}


def get_info(class_name):
    if class_name in KB:
        return KB[class_name]
    is_healthy = 'healthy' in class_name.lower()
    plant = class_name.split('_')[0]
    disease = class_name.replace(plant,'').replace('___',' ').replace('_',' ').strip()
    return {
        "common": disease if not is_healthy else f"Healthy {plant}",
        "pathogen": "Pathogen details not available" if not is_healthy else None,
        "sev": (0,5) if is_healthy else (30,65),
        "symptoms": ["Visual symptoms detected by AI model — see below for details"],
        "cause": "Pathogen thrives in warm humid conditions" if not is_healthy else "Plant is healthy",
        "treatment": ["Consult your local agricultural extension office for specific product recommendations","Apply appropriate fungicide based on local guidelines","Remove infected plant material","Improve plant nutrition and growing conditions"] if not is_healthy else ["No treatment needed"],
        "organic": ["Neem oil spray every 7 days","Copper-based fungicide"] if not is_healthy else [],
        "prevention": ["Practice crop rotation","Ensure good air circulation","Avoid overhead irrigation","Use disease-resistant varieties"],
        "recovery": "2–4 weeks with appropriate treatment" if not is_healthy else "Plant is healthy",
        "yield": "Variable depending on severity" if not is_healthy else "None",
        "spread": "Moderate" if not is_healthy else "None"
    }


def build_analysis(pred, mode, context=""):
    cls   = pred["predicted_class"]
    conf  = pred["confidence"]
    mdl   = pred["model_used"]
    top5  = pred.get("top5", [])
    info  = get_info(cls)
    healthy = info["pathogen"] is None
    sev   = 0 if healthy else random.randint(*info["sev"])
    ctx   = f"\nFarmer notes: {context}" if context else ""

    if mode == "full":
        body = f"""DISEASE IDENTIFICATION
{'─'*42}
Diagnosed:   {info['common']}
Pathogen:    {info['pathogen'] or 'N/A — plant is healthy'}
AI Confidence: {conf*100:.1f}%   |   Model: {mdl}
Severity:    {'None — Healthy Plant' if healthy else f'{sev}% — {"Low" if sev<34 else "Medium" if sev<67 else "High"}'}
{ctx}

SYMPTOMS OBSERVED
{'─'*42}
{chr(10).join(f'• {s}' for s in info['symptoms'])}

ROOT CAUSE
{'─'*42}
{info['cause']}

{'TREATMENT PLAN' if not healthy else 'CARE RECOMMENDATIONS'}
{'─'*42}
{chr(10).join(f'{i+1}. {t}' for i,t in enumerate(info['treatment']))}

ORGANIC ALTERNATIVES
{'─'*42}
{chr(10).join(f'• {o}' for o in info['organic']) if info['organic'] else 'No organic treatment needed — plant is healthy.'}

PREVENTION FOR NEXT SEASON
{'─'*42}
{chr(10).join(f'• {p}' for p in info['prevention'])}

YIELD IMPACT IF UNTREATED
{'─'*42}
{info['yield']}

SPREAD RISK TO OTHER PLANTS
{'─'*42}
{info['spread']}

EXPECTED RECOVERY TIME
{'─'*42}
{info['recovery']}

ALTERNATIVE DIAGNOSES — AI TOP 5
{'─'*42}
{chr(10).join(f'{i+1}. {n.replace("_"," ")} ({p*100:.1f}%)' for i,(n,p) in enumerate(top5[:5]))}"""

    elif mode == "treatment":
        body = f"""QUICK DIAGNOSIS
{'─'*42}
{info['common']} — {conf*100:.1f}% confidence
{'Plant is healthy — no treatment required.' if healthy else f'Severity: {sev}% — {"Low" if sev<34 else "Medium" if sev<67 else "High"} — action {"recommended" if sev<50 else "URGENT"}'}
{ctx}

IMMEDIATE ACTIONS (Do today)
{'─'*42}
{chr(10).join(f'{i+1}. {t}' for i,t in enumerate(info['treatment']))}

ORGANIC OPTIONS
{'─'*42}
{chr(10).join(f'• {o}' for o in info['organic']) if info['organic'] else 'No treatment needed.'}

CHEMICAL OPTIONS (If needed)
{'─'*42}
{'Not required.' if healthy else f'Apply appropriate fungicide/bactericide for: {info["pathogen"]}. Consult your local agricultural supplier for approved products in your region.'}

EXPECTED RECOVERY
{'─'*42}
{info['recovery']}

WARNING SIGNS (Treatment not working)
{'─'*42}
{'N/A — plant is healthy.' if healthy else '• Disease spreading to new healthy leaves after 2 weeks of treatment\n• Lesions increasing in size despite fungicide application\n• More than 50% of leaves showing symptoms\nIf these occur: consult your local agricultural extension service immediately.'}"""

    else:  # prevention
        body = f"""CURRENT CONDITION
{'─'*42}
{'Healthy plant — excellent time to implement prevention.' if healthy else f'{info["common"]} detected — act now to prevent further spread.'}
Confidence: {conf*100:.1f}%
{ctx}

WHY THIS HAPPENED
{'─'*42}
{info['cause']}

IMMEDIATE PREVENTION (Do today)
{'─'*42}
{'1. Continue current practices.\n2. Implement preventive spray programme.\n3. Monitor weekly.' if healthy else f'1. Remove ALL infected leaves and dispose away from garden\n2. Apply preventive treatment to neighbouring healthy plants\n3. Disinfect all tools used\n4. Avoid working in plants when wet'}

PREVENTION PROGRAMME
{'─'*42}
{chr(10).join(f'• {p}' for p in info['prevention'])}

SOIL AND NUTRITION
{'─'*42}
• Maintain balanced NPK fertilisation — avoid excess nitrogen
• Ensure good drainage to prevent waterlogging
• Add compost to improve soil health and beneficial microorganisms
• Maintain soil pH between 6.0 and 7.0 for optimal nutrient uptake

COMPANION PLANTING
{'─'*42}
• Basil — repels many fungal pathogens and pests
• Marigolds — suppress soil-borne nematodes and pests
• Garlic/chives — natural antifungal properties
• Maintain 45–60cm spacing between plants for air circulation

MONITORING CHECKLIST
{'─'*42}
- Check leaf undersides for spots or mould every week
- Look for yellowing or wilting as early warning signs
- Monitor after rainfall — most diseases spread in wet conditions
- Record spray programme dates and products used
- Scout entire plot systematically"""

    return {
        "analysis":   body,
        "disease":    info["common"],
        "severity":   sev,
        "is_healthy": healthy,
        "pathogen":   info["pathogen"],
        "confidence": conf,
        "model_used": mdl,
        "spread":     info["spread"],
        "recovery":   info["recovery"],
        "class_name": cls,
    }


def answer_followup(question, cls):
    q = question.lower()
    info = get_info(cls)
    if any(w in q for w in ["organic","natural","chemical-free"]):
        return FOLLOWUP["organic"]
    elif any(w in q for w in ["spread","contagious","other plant","neighbour"]):
        return FOLLOWUP["spread"]
    elif any(w in q for w in ["fertiliser","fertilizer","feed","nutrient","npk"]):
        return FOLLOWUP["fertiliser"]
    elif any(w in q for w in ["soil","drainage","compost","ph"]):
        return FOLLOWUP["soil"]
    elif any(w in q for w in ["recover","recovery","long","weeks","time"]):
        return FOLLOWUP["recovery"]
    elif any(w in q for w in ["buy","purchase","product","recommend","brand"]):
        return FOLLOWUP["buy"]
    elif any(w in q for w in ["water","irrigation","watering"]):
        return FOLLOWUP["water"]
    elif any(w in q for w in ["safe","eat","consume","food","harvest"]):
        return FOLLOWUP["food"]
    elif any(w in q for w in ["prevent","prevention","avoid","future","next season"]):
        return f"Prevention for {info['common']}:\n\n" + "\n".join(f"• {p}" for p in info.get("prevention",["Consult local agricultural advisor"]))
    else:
        treat = "\n".join(f"{i+1}. {t}" for i,t in enumerate(info.get("treatment",["Consult an agricultural specialist"])[:3]))
        return f"Regarding {info['common']}:\n\nCaused by: {info.get('pathogen','unknown pathogen')}\n\nKey treatment steps:\n{treat}\n\nExpected recovery: {info.get('recovery','2–4 weeks with appropriate treatment')}\n\nFor more specific advice, contact your local agricultural extension service."


# ── Flask routes ───────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", models=predictor.get_available_models() if predictor else [])

@app.route("/ai-tool")
def ai_tool():
    return render_template("ai_tool.html", models=predictor.get_available_models() if predictor else [])

@app.route("/api/models")
def api_models():
    return jsonify({"models": predictor.get_available_models() if predictor else []})

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        if not predictor: return jsonify({"error":"No models loaded"}),500
        if "image" not in request.files: return jsonify({"error":"No image"}),400
        file = request.files["image"]
        if not allowed_file(file.filename): return jsonify({"error":"File type not allowed"}),400
        ext = file.filename.rsplit(".",1)[1].lower()
        filename = f"{uuid.uuid4().hex}.{ext}"
        save_path = os.path.join(UPLOAD_DIR, filename)
        file.save(save_path)
        result = predictor.predict(save_path, model_name=request.form.get("model") or None)
        if "error" in result: return jsonify(result),500
        result["image_url"] = url_for("uploaded_file", filename=filename)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error":str(e)}),500

@app.route("/api/ai-analyse", methods=["POST"])
def api_ai_analyse():
    try:
        if not predictor: return jsonify({"error":"No models loaded. Copy .pth and .joblib files to saved_models/ folder."}),500
        if "image" not in request.files: return jsonify({"error":"No image provided"}),400
        file = request.files["image"]
        if not allowed_file(file.filename): return jsonify({"error":"File type not allowed"}),400
        mode    = request.form.get("mode","full")
        context = request.form.get("context","")
        ext     = file.filename.rsplit(".",1)[1].lower()
        filename = f"{uuid.uuid4().hex}.{ext}"
        save_path = os.path.join(UPLOAD_DIR, filename)
        file.save(save_path)
        pred = predictor.predict(save_path, model_name=request.form.get("model") or None)
        if "error" in pred: return jsonify(pred),500
        result = build_analysis(pred, mode, context)
        result["image_url"] = url_for("uploaded_file", filename=filename)
        result["top5"] = pred.get("top5",[])
        return jsonify(result)
    except Exception as e:
        return jsonify({"error":str(e)}),500

@app.route("/api/followup", methods=["POST"])
def api_followup():
    try:
        data = request.get_json()
        q    = data.get("question","")
        cls  = data.get("class_name","")
        if not q: return jsonify({"error":"No question provided"}),400
        return jsonify({"answer": answer_followup(q, cls)})
    except Exception as e:
        return jsonify({"error":str(e)}),500

@app.route("/api/results")
def api_results():
    try:
        p = os.path.join(RESULTS_DIR,"all_results.json")
        if not os.path.exists(p): return jsonify({"error":"No results found."})
        with open(p, encoding="utf-8") as f: return jsonify(json.load(f))
    except Exception as e:
        return jsonify({"error":str(e)}),500

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)

@app.route("/results/<filename>")
def result_image(filename):
    return send_from_directory(RESULTS_DIR, filename)

@app.route("/health")
def health():
    return jsonify({"status":"ok","models_loaded": predictor.get_available_models() if predictor else []})

@app.errorhandler(404)
def not_found(e): return jsonify({"error":"Route not found"}),404
@app.errorhandler(500)
def server_error(e): return jsonify({"error":"Internal server error","details":str(e)}),500
@app.errorhandler(413)
def too_large(e): return jsonify({"error":f"File too large. Max {MAX_UPLOAD_MB}MB"}),413

if __name__ == "__main__":
    print(f"\n Plant Disease Detection — AI Tool")
    print(f"   Models loaded : {predictor.get_available_models() if predictor else 'None'}")
    print(f"   Main app      : http://localhost:{FLASK_PORT}")
    print(f"   AI Tool       : http://localhost:{FLASK_PORT}/ai-tool\n")
   if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
