# Demo: top-5 per model

Qualitative output of `make demo`. Not an evaluation.


## Query 1: "Something red that isn't too heavy, a bit fruity and easy to drink"


**popularity**

| # | Wine | Variety | Country | Price | Points | Tasting note (start) |
|---|---|---|---|---|---|---|
| 1 | Wines & Winemakers 2009 Aguia Moura em Vinhas Velhas Reserva Red (Douro) | Portuguese Red | Portugal | $18 | 93 | This is a smooth, impressively rich wine powered by silky tannins and ripe black fruits. The palate is intensely concentrated with a big, bold structure … |
| 2 | Wines & Winemakers 2012 Companhia das Lezírias Tyto Alba Vinhas Protegidas Touriga Nacional (Tejo) | Touriga Nacional | Portugal | $36 | 93 | Named after the barn owl (Tyto alba) that nests in the woods of the Lezirias estate, this has great tannins that give shape to the … |
| 3 | Wines & Winemakers 2014 Companhia das Lezirias Tyto Alba Touriga Nacional (Tejo) | Touriga Nacional | Portugal | $36 | 93 | This 100% Touriga Nacional comes from the government-owned estate in the north of Tejo. Named after the barn owl that is found in the woods, … |
| 4 | Wines & Winemakers 2015 Companhia das Lezirias 1836 Grande Reserva Branco Fernão Pires (Tejo) | Fernão Pires | Portugal | $77 | 93 | This fine low-production wood-aged wine is almost Burgundian in its creamy richness. The toast is nicely judged giving a wine that has just enough acidity … |
| 5 | Wines & Winemakers 2012 Pegos Claros Vinhas Velhas Reserva Castelão (Palmela) | Castelão | Portugal | $18 | 93 | A serious and dense wine, this is rich and full of dark juicy fruit. It has plenty of structure of course, with its rich tannins … |

**tfidf**

| # | Wine | Variety | Country | Price | Points | Tasting note (start) |
|---|---|---|---|---|---|---|
| 1 | Valley View 2000 Anna Maria Chardonnay (Oregon) | Chardonnay | US | $15 | 87 | Lots of ripe apple and tropical fruits, lots of butterscotch and caramel flavors, but not too heavy. Good balance, good acids, a very appealing, forward … |
| 2 | Kenefick Ranch 2007 Caitlin's Select Cabernet Franc (Calistoga) | Cabernet Franc | US | $52 | 86 | Almost too ripe and fruity, with jammy, pie-filling blackberry and cherry flavors. Lots of oak, too, giving the wine a caramelized sweetness. A bit heavy-handed, … |
| 3 | Jose Maria Vieira 2014 Borges Quinta da Soalheira Red (Douro) | Portuguese Red | Portugal | $8 | 85 | This is an easy, fruity wine. Soft and light in character, it has attractive red fruits and acidity. Drink this wine now. |
| 4 | Belle Glos 2010 Meiomi Pinot Noir (Monterey-Santa Barbara-Sonoma) | Pinot Noir | US | $22 | 86 | A pretty good Pinot Noir, elegantly silky as it should be, with flavors of tart cherries, cola, red licorice and sandalwood. It's a bit too … |
| 5 | Casal da Coelheira 2014 Rosé (Tejo) | Rosé | Portugal | $12 | 81 | This electric pink colored wine is both too rich and too heavy. It only survives because of a fresh streak of acidity that comes through … |

**attribute**

| # | Wine | Variety | Country | Price | Points | Tasting note (start) |
|---|---|---|---|---|---|---|
| 1 | Tenuta dell'Ornellaia 2007 Masseto Merlot (Toscana) | Merlot | Italy | $460 | 100 | A perfect wine from a classic vintage, the 2007 Masseto (100% Merlot from a 17-acre vineyard of the same name) opens with an unabashedly opulent … |
| 2 | Casa Ferreirinha 2008 Barca-Velha Red (Douro) | Portuguese Red | Portugal | $450 | 100 | This is the latest release of what has long been regarded as Portugal's iconic wine. And it is magnificent. The last vintage was the 2004 … |
| 3 | Biondi Santi 2010 Riserva  (Brunello di Montalcino) | Sangiovese | Italy | $550 | 100 | This gorgeous, fragrant wine opens with classic Sangiovese scents of violet, rose, perfumed red berry, new leather and a whiff of baking spice. The elegant, … |
| 4 | Cardinale 2006 Cabernet Sauvignon (Napa Valley) | Cabernet Sauvignon | US | $200 | 100 | Tasted in a flight of great and famous Napa wines, this Cardinale stood at the head of the pack. Starts with a very fine nose … |
| 5 | Château Léoville Barton 2010  Saint-Julien | Bordeaux-style Red Blend | France | $150 | 100 | This is a magnificently solid wine, initially even a bit severe. At this young stage, the tannins dominate, but it's also full of black fruit … |

**semantic**

| # | Wine | Variety | Country | Price | Points | Tasting note (start) |
|---|---|---|---|---|---|---|
| 1 | Domaine de Pellehaut 2008 Harmonie de Gascogne Red (Vin de Pays des Côtes de Gascogne) | Red Blend | France | – | 84 | Lightweight red from Gascony, with a touch of tannin, red berry fruits and crisp acidity. A red wine with an attractive bite. |
| 2 | Grace Lane 2012 Cabernet Sauvignon (Columbia Valley (WA)) | Cabernet Sauvignon | US | $14 | 86 | An all-purpose red, it has pretty fruit scents and flavors of strawberry and cherry candy. There's also a suggestion of café latté and a hint … |
| 3 | Schulz 2006 Samuel's Cuveé Red (Napa Valley) | Red Blend | US | $25 | 87 | A pleasant, easy-drinking red, with some real elegance. Tastes very extracted in berry, currant and chocolate flavors, almost sweet like a pastry, but turns nicely … |
| 4 | Rovisco Garcia 2010 Tinto Red (Alentejano) | Portuguese Red | Portugal | $25 | 87 | A fruity style of red, it is made to be drunk young, possibly even lightly chilled. It has soft tannins plus a fine balance between … |
| 5 | Cave B 2011 Sangiovese (Columbia Valley (WA)) | Sangiovese | US | $33 | 86 | Lightly flavored with red fruits, it's pleasant but rather generic. It carries some alcohol heat through a thin finish. |

**semantic_bge**

| # | Wine | Variety | Country | Price | Points | Tasting note (start) |
|---|---|---|---|---|---|---|
| 1 | Domaine Michel Fonne 2013 Tradition Pinot Noir (Alsace) | Pinot Noir | France | $17 | 86 | Pure red fruit seems to have a bruised apple edge. The palate is simple and easy-drinking and despite its pretty ruby color comes across as … |
| 2 | Cinnabar 2011 Tripartite Red (California) | Red Blend | US | $42 | 86 | A lighter style of red blend, this is easy to drink and enjoy. It doesn't have a strong personality but is well-balanced, fruity and juicy, … |
| 3 | Laroque 2015 Pinot Noir (Vin de Pays Cité de Carcassonne) | Pinot Noir | France | $14 | 84 | Notes of red currant, rhubarb, tart strawberry and Cherry Dr. Pepper grace the nose and mouth of this straightforward and easy-drinking red. It's lightweight and … |
| 4 | Schulz 2006 Samuel's Cuveé Red (Napa Valley) | Red Blend | US | $25 | 87 | A pleasant, easy-drinking red, with some real elegance. Tastes very extracted in berry, currant and chocolate flavors, almost sweet like a pastry, but turns nicely … |
| 5 | Quinta da Alorna 2009 What's for Dinner? Tinto Red (Tejo) | Portuguese Red | Portugal | $8 | 85 | Soft, easy drinking. This red has berry fruits, just the right amount of dusty and juicy flavors. Screwcap. |

## Query 2: "A fresh white that tastes like lemon and green apple, not sweet"


**popularity**

| # | Wine | Variety | Country | Price | Points | Tasting note (start) |
|---|---|---|---|---|---|---|
| 1 | Wines & Winemakers 2009 Aguia Moura em Vinhas Velhas Reserva Red (Douro) | Portuguese Red | Portugal | $18 | 93 | This is a smooth, impressively rich wine powered by silky tannins and ripe black fruits. The palate is intensely concentrated with a big, bold structure … |
| 2 | Wines & Winemakers 2012 Companhia das Lezírias Tyto Alba Vinhas Protegidas Touriga Nacional (Tejo) | Touriga Nacional | Portugal | $36 | 93 | Named after the barn owl (Tyto alba) that nests in the woods of the Lezirias estate, this has great tannins that give shape to the … |
| 3 | Wines & Winemakers 2014 Companhia das Lezirias Tyto Alba Touriga Nacional (Tejo) | Touriga Nacional | Portugal | $36 | 93 | This 100% Touriga Nacional comes from the government-owned estate in the north of Tejo. Named after the barn owl that is found in the woods, … |
| 4 | Wines & Winemakers 2015 Companhia das Lezirias 1836 Grande Reserva Branco Fernão Pires (Tejo) | Fernão Pires | Portugal | $77 | 93 | This fine low-production wood-aged wine is almost Burgundian in its creamy richness. The toast is nicely judged giving a wine that has just enough acidity … |
| 5 | Wines & Winemakers 2012 Pegos Claros Vinhas Velhas Reserva Castelão (Palmela) | Castelão | Portugal | $18 | 93 | A serious and dense wine, this is rich and full of dark juicy fruit. It has plenty of structure of course, with its rich tannins … |

**tfidf**

| # | Wine | Variety | Country | Price | Points | Tasting note (start) |
|---|---|---|---|---|---|---|
| 1 | Finca Las Moras 2007 Reserve Chardonnay (San Juan) | Chardonnay | Argentina | $12 | 85 | Not exactly a world-class white wine but at least it smells and tastes like Chardonnay. The nose offers some butter and lemon curd, while the … |
| 2 | Michael Pozzan 2012 Dante Reserve Pinot Grigio (Paso Robles) | Pinot Grigio | US | $10 | 82 | Tastes sweet, with citrus, white flower and green pea notes. |
| 3 | Valley of the Moon 2010 Pinot Blanc (Sonoma County) | Pinot Blanc | US | $14 | 86 | Tastes like a Chardonnay, with tropical fruit, peach and green apple flavors wrapped into buttercream and vanilla. Ripe and sweet, but balanced by crisp acidity. … |
| 4 | Karma NV Side View Chardonnay (North Coast) | Chardonnay | US | $15 | 87 | Like a liquid lemon meringue pie, this Chardonnay has bright fruit flavors, lemony acidity and a creamy finish. It smells like ripe apple and pineapple, … |
| 5 | Archgate Cellars 2015 Reserve Bottling Pinot Grigio (Lodi) | Pinot Grigio | US | $12 | 85 | This wine tastes very tangy and crisp. Medium bodied, with lemon and green-apple flavors, it's refreshing and mouth-cleansing. |

**attribute**

| # | Wine | Variety | Country | Price | Points | Tasting note (start) |
|---|---|---|---|---|---|---|
| 1 | Salon 2006 Le Mesnil Blanc de Blancs Brut Chardonnay (Champagne) | Chardonnay | France | $617 | 100 | This new release from a great vintage for Chardonnay in Champagne is a perfect blanc de blancs, showing a wealth of intensity from the crisp, … |
| 2 | Château Haut-Brion 2014  Pessac-Léognan | Bordeaux-style White Blend | France | $848 | 100 | Full of ripe fruit, opulent and concentrated, this is a fabulous and impressive wine. It has a beautiful line of acidity balanced with ripe fruits. … |
| 3 | Krug 2002 Clos du Mesnil Brut Blanc de Blancs Chardonnay (Champagne) | Chardonnay | France | $800 | 99 | Pure Chardonnay from the walled vineyard in the heart of Le Mesnil, this intense wine from a great vintage is tight and mineral, still showing … |
| 4 | Domaine Huët 2009 Cuvée Constance 500ml  (Vouvray) | Chenin Blanc | France | $159 | 99 | Even more concentrated than Huët's superb 2009 Première Trie, this is outstanding. It has both great concentration and elegance. The wine is touched by botrytis, … |
| 5 | Domaine Leflaive 2010  Bâtard-Montrachet | Chardonnay | France | $560 | 99 | In this great year for white Burgundy, Bâtard-Montrachet shines out as the epitome of balance between superb fruit and intense mineral and terroir characters. Its … |

**semantic**

| # | Wine | Variety | Country | Price | Points | Tasting note (start) |
|---|---|---|---|---|---|---|
| 1 | Fat Barrel 2015 Bob's Chenin Blanc (South Africa) | Chenin Blanc | South Africa | $9 | 84 | A lightweight and easygoing white, this offers soft aromas and flavors of green apple, melon rind and citrus, all framed by bright acidity. The finish … |
| 2 | Gilbert Cellars 2008 Unoaked Chardonnay (Wahluke Slope) | Chardonnay | US | $17 | 83 | Light, watery and thin, with simple lemon and green apple flavors, and plenty of acidity. |
| 3 | Heron Hill 2015 Semi-Sweet Riesling (Finger Lakes) | Riesling | US | $15 | 87 | Fresh lemon, grapefruit and green apple build in ripeness and sweetness on the palate of this light-bodied Riesling. It's a juicy, uncomplicated white but offers … |
| 4 | Heron Hill 2015 Unaoked Chardonnay (Finger Lakes) | Chardonnay | US | $15 | 86 | Fresh lemon and red-apple flavors are bright and clean in this medium-bodied dry white. While brisk in texture and approachably fresh, the finish is marked … |
| 5 | The Calling 2013 Jewell Vineyard Dutton Ranch Chardonnay (Russian River Valley) | Chardonnay | US | $51 | 88 | This fleshy white has a tropical ripeness that almost appears sweet, set against a prodigious use of caramelized oak. Tart lemon and apple complement in … |

**semantic_bge**

| # | Wine | Variety | Country | Price | Points | Tasting note (start) |
|---|---|---|---|---|---|---|
| 1 | Arauco 2011 Reserva Pinot Grigio (Central Valley) | Pinot Grigio | Chile | $14 | 83 | Minerally and neutral on the nose, this tastes fresh and crisp like a green apple, with dry flavors of white fruit and bitter almond. It … |
| 2 | Corte Adami 2016  Soave | White Blend | Italy | $13 | 85 | This extremely simple but pleasantly fresh white offers green apple, peach and citrus hints. Zesty acidity gives it a lively character. |
| 3 | Oak Grove 2013 Family Reserve Pinot Grigio (California) | Pinot Grigio | US | $8 | 85 | It tastes lean and refreshing, with light lemon and green apple in the aroma and flavors, rather light body and a slightly tangy finish. It … |
| 4 | Undurraga 2008 Aliwen Reserva Sauvignon Blanc (Central Valley) | Sauvignon Blanc | Chile | $12 | 86 | Fresh but not overly racy, with apple and pear aromas. There's ample snap and pop to the palate, but not a whole lot of body … |
| 5 | Heron Hill 2015 Semi-Sweet Riesling (Finger Lakes) | Riesling | US | $15 | 87 | Fresh lemon, grapefruit and green apple build in ripeness and sweetness on the palate of this light-bodied Riesling. It's a juicy, uncomplicated white but offers … |

## Query 3: "Something bubbly and light to celebrate with friends"


**popularity**

| # | Wine | Variety | Country | Price | Points | Tasting note (start) |
|---|---|---|---|---|---|---|
| 1 | Wines & Winemakers 2009 Aguia Moura em Vinhas Velhas Reserva Red (Douro) | Portuguese Red | Portugal | $18 | 93 | This is a smooth, impressively rich wine powered by silky tannins and ripe black fruits. The palate is intensely concentrated with a big, bold structure … |
| 2 | Wines & Winemakers 2012 Companhia das Lezírias Tyto Alba Vinhas Protegidas Touriga Nacional (Tejo) | Touriga Nacional | Portugal | $36 | 93 | Named after the barn owl (Tyto alba) that nests in the woods of the Lezirias estate, this has great tannins that give shape to the … |
| 3 | Wines & Winemakers 2014 Companhia das Lezirias Tyto Alba Touriga Nacional (Tejo) | Touriga Nacional | Portugal | $36 | 93 | This 100% Touriga Nacional comes from the government-owned estate in the north of Tejo. Named after the barn owl that is found in the woods, … |
| 4 | Wines & Winemakers 2015 Companhia das Lezirias 1836 Grande Reserva Branco Fernão Pires (Tejo) | Fernão Pires | Portugal | $77 | 93 | This fine low-production wood-aged wine is almost Burgundian in its creamy richness. The toast is nicely judged giving a wine that has just enough acidity … |
| 5 | Wines & Winemakers 2012 Pegos Claros Vinhas Velhas Reserva Castelão (Palmela) | Castelão | Portugal | $18 | 93 | A serious and dense wine, this is rich and full of dark juicy fruit. It has plenty of structure of course, with its rich tannins … |

**tfidf**

| # | Wine | Variety | Country | Price | Points | Tasting note (start) |
|---|---|---|---|---|---|---|
| 1 | V. Sattui 2011 Morisoli Vineyard Cabernet Sauvignon (Napa Valley) | Cabernet Sauvignon | US | $65 | 83 | Preponderant green, herbal aroma and flavor marks this wine. The mix of pencil lead and a smoky char never really make friends. |
| 2 | Josef Brigl 2007 Sielo Blu Pinot Grigio (Alto Adige) | Pinot Grigio | Italy | $13 | 86 | Passion fruit and citrus describe the nose of this crisp but simple Pinot Grigio. The wine is luminous and light and leaves a long citrusy … |
| 3 | Monteverro 2011 Il Nostro Vermentino (Toscana) | Vermentino | Italy | – | 88 | Very crisp, clean and polished, this is an easy white wine to enjoy with friends and grilled fish. It's redolent of cut grass, citrus, peach … |
| 4 | Cusumano 2011 Syrah (Sicilia) | Syrah | Italy | $12 | 87 | Plump and meaty, this opens with dark aromas of blackberry, spice, crushed pepper and bresaola. This would drink well at informal occasions with family and … |
| 5 | Mosby NV Stelline di Moscato Orange Muscat (Santa Barbara County) | Orange Muscat | US | $18 | 87 | A good, rich, dry bubbly, with a creamy, leesy taste and long finish. The flavors of figgy vanilla and praline are unusual for a bubbly, … |

**attribute**

| # | Wine | Variety | Country | Price | Points | Tasting note (start) |
|---|---|---|---|---|---|---|
| 1 | Krug 2002 Brut  (Champagne) | Champagne Blend | France | $259 | 100 | This is a fabulous wine from the greatest Champagne vintage so far this century. Still young, it blends almost equal amounts of Pinot Noir and … |
| 2 | Louis Roederer 2008 Cristal Vintage Brut  (Champagne) | Champagne Blend | France | $250 | 100 | This latest incarnation of the famous brand is a superb wine. It is on par with, maybe even better than, the already legendary 2002. Its … |
| 3 | Iron Horse 2003 Joy! Estate Bottled Sparkling (Green Valley) | Sparkling Blend | US | $250 | 98 | This is an incredibly beautiful and yes, joyful, sparkler, made from equal parts Chardonnay and Pinot Noir. This particular offering was disgorged in July 2016. … |
| 4 | Schramsberg 2004 Reserve Sparkling (Napa-Mendocino-Sonoma-Marin) | Sparkling Blend | US | $110 | 98 | The opulence of this sparkling wine has to be tasted to be believed. It's far softer and silkier than the winery's companion new release, the … |
| 5 | Schramsberg 2004 J. Schram Rosé Sparkling (North Coast) | Sparkling Blend | US | $130 | 98 | An awesome rosé, just a tremendous feat. Made from 11 vineyards in 4 counties, it shows a masterful art of the blend. Insanely rich in … |

**semantic**

| # | Wine | Variety | Country | Price | Points | Tasting note (start) |
|---|---|---|---|---|---|---|
| 1 | Bota Box 2011 Pinot Grigio (California) | Pinot Grigio | US | $20 | 85 | Throwing a party? Need plenty of white wine to make the rounds? This is the solution. With slightly sweet apricot, orange and vanilla flavors, it's … |
| 2 | Raymond 2009 R Collection Lot No 7 Field Blend Red (California) | Red Blend | US | $15 | 83 | A simple, slightly sweet wine, with candied berry, sugar and spice flavors. Drink it at picnics, parties, the beach. |
| 3 | Winzer Krems 2012 Haus Österrich Riesling (Österreichischer Sekt) | Riesling | Austria | $18 | 87 | A bit of apple, a bit of citrus, a bit of toast—things are just hinted at in this frothy sparkler which would make an ideal, … |
| 4 | Gloria Ferrer NV VA de VI Ultra Cuvée Sparkling (Sonoma County) | Sparkling Blend | US | $22 | 87 | A nice everyday sparkler. It's a little sweet and very forward in lime and strawberry flavors, with the yeast and toast complications of a fine … |
| 5 | Trinchero 2007 Cabernet Franc (Napa Valley) | Cabernet Franc | US | $50 | 87 | A good wine, dry and tannic, but a little one-dimensional in richness. Shows pretty flavors of red and black cherries, with a hint of mint … |

**semantic_bge**

| # | Wine | Variety | Country | Price | Points | Tasting note (start) |
|---|---|---|---|---|---|---|
| 1 | Gloria Ferrer NV Sonoma Brut Sparkling (Sonoma County) | Sparkling Blend | US | $20 | 86 | This bubbly is rich in cherry and raspberry fruit, in addition to the usual sparkling wine notes of bread dough and smoke. It's a little … |
| 2 | Korbel NV Sweet Cuvée Sparkling (California) | Sparkling Blend | US | $13 | 84 | This bubbly is on the sweet side. It has sugary pineapple and orange flavors. A little scoury in texture, it's an easy sparkling wine to … |
| 3 | Domaine Laurier NV Brut Sparkling (California) | Sparkling Blend | US | $13 | 85 | Tight in texture and rich in vanilla bean, white sugar and lemon-curd flavors, this is a simple bubbly. For the price, it's fancy enough to … |
| 4 | Cà Vittoria 2007 Millesimato Dry  (Prosecco di Conegliano e Valdobbiadene) | Prosecco | Italy | $20 | 86 | This is a sweet, floral bubbly with mature notes of pear and peach and chewy sweetness. It has a soft, creamy texture and would pair … |
| 5 | Mumm Napa 2008 Blanc de Blancs Sparkling (Napa Valley) | Sparkling Blend | US | $38 | 89 | Made from Chardonnay and Pinot Gris, this bubbly is light in body and elegant, with a cloud-like lightness. The palate is high toned with flavors … |

## Query 4: "A sweet wine that tastes like honey and ripe peaches, for dessert"


**popularity**

| # | Wine | Variety | Country | Price | Points | Tasting note (start) |
|---|---|---|---|---|---|---|
| 1 | Wines & Winemakers 2009 Aguia Moura em Vinhas Velhas Reserva Red (Douro) | Portuguese Red | Portugal | $18 | 93 | This is a smooth, impressively rich wine powered by silky tannins and ripe black fruits. The palate is intensely concentrated with a big, bold structure … |
| 2 | Wines & Winemakers 2012 Companhia das Lezírias Tyto Alba Vinhas Protegidas Touriga Nacional (Tejo) | Touriga Nacional | Portugal | $36 | 93 | Named after the barn owl (Tyto alba) that nests in the woods of the Lezirias estate, this has great tannins that give shape to the … |
| 3 | Wines & Winemakers 2014 Companhia das Lezirias Tyto Alba Touriga Nacional (Tejo) | Touriga Nacional | Portugal | $36 | 93 | This 100% Touriga Nacional comes from the government-owned estate in the north of Tejo. Named after the barn owl that is found in the woods, … |
| 4 | Wines & Winemakers 2015 Companhia das Lezirias 1836 Grande Reserva Branco Fernão Pires (Tejo) | Fernão Pires | Portugal | $77 | 93 | This fine low-production wood-aged wine is almost Burgundian in its creamy richness. The toast is nicely judged giving a wine that has just enough acidity … |
| 5 | Wines & Winemakers 2012 Pegos Claros Vinhas Velhas Reserva Castelão (Palmela) | Castelão | Portugal | $18 | 93 | A serious and dense wine, this is rich and full of dark juicy fruit. It has plenty of structure of course, with its rich tannins … |

**tfidf**

| # | Wine | Variety | Country | Price | Points | Tasting note (start) |
|---|---|---|---|---|---|---|
| 1 | Coastal Ridge 2010 Riesling (California) | Riesling | US | $9 | 83 | If you like your white wines fruity and sweet, this one's for you. It has pretty flavors of peaches, honey and spices. |
| 2 | Bota Box 2014 Riesling (California) | Riesling | US | $23 | 83 | Lightly sweet and simply fruity, this wine tastes a bit like peaches and has a gentle texture. |
| 3 | Montevina 2006 Pinot Grigio (Amador County) | Pinot Grigio | US | $10 | 82 | Made in the sweet, sugary style that many consumers like, this is almost like an off-dry dessert wine. Tastes like a citrus and peach power … |
| 4 | Molnar Family 2009 Poseidon's Vineyard Chardonnay (Carneros) | Chardonnay | US | $25 | 83 | A simple, sweet wine, with fruit juice flavors of pineapple, peaches, apples and honey. |
| 5 | Fanucchi 2005 Wood Road Vineyard Trousseau Gris (Russian River Valley) | Trousseau Gris | US | $19 | 84 | Smells honeyed and tastes like one of those fruity bottled sodas. It's a riot of peaches, pears and vanilla cream, with a honey-sweet finish. |

**attribute**

| # | Wine | Variety | Country | Price | Points | Tasting note (start) |
|---|---|---|---|---|---|---|
| 1 | Chambers Rosewood Vineyards NV Rare Muscat (Rutherglen) | Muscat | Australia | $350 | 100 | This wine contains some material over 100 years old, but shows no signs of fragility. Instead, it's concentrated through age and should hold in the … |
| 2 | Avignonesi 1995 Occhio di Pernice  (Vin Santo di Montepulciano) | Prugnolo Gentile | Italy | $210 | 100 | Thick as molasses and dark as caramelized brown sugar, the wine oozes out of the bottle releasing concentrated aromas of butterscotch, toffee, honey, licorice, coffee, … |
| 3 | Krug 2002 Brut  (Champagne) | Champagne Blend | France | $259 | 100 | This is a fabulous wine from the greatest Champagne vintage so far this century. Still young, it blends almost equal amounts of Pinot Noir and … |
| 4 | Tenuta dell'Ornellaia 2007 Masseto Merlot (Toscana) | Merlot | Italy | $460 | 100 | A perfect wine from a classic vintage, the 2007 Masseto (100% Merlot from a 17-acre vineyard of the same name) opens with an unabashedly opulent … |
| 5 | Casa Ferreirinha 2008 Barca-Velha Red (Douro) | Portuguese Red | Portugal | $450 | 100 | This is the latest release of what has long been regarded as Portugal's iconic wine. And it is magnificent. The last vintage was the 2004 … |

**semantic**

| # | Wine | Variety | Country | Price | Points | Tasting note (start) |
|---|---|---|---|---|---|---|
| 1 | Jana 2007 Old Vine Riesling (Napa Valley) | Riesling | US | $20 | 85 | You could classify this as an off-dry wine, but it's just sweet and honeyed enough to be a dessert wine. With pleasant acidity providing balance, … |
| 2 | Totally Random NV Peach White (California) | White Blend | US | $10 | 82 | This peach-flavored wine is sweet in honey and brisk in acidity. |
| 3 | Domaine d'Arton 2015 Les Deux Cèdres Gros Manseng (Côtes de Gascogne) | Gros Manseng | France | – | 87 | There is an appealing touch of honey in this medium-sweet wine. It has a fine balance between acidity and ripe apricot and peach fruit. It … |
| 4 | Toulouse 2013 Helluva Vineyard Pinot Gris (Anderson Valley) | Pinot Gris | US | $24 | 90 | Fresh, crisp and uncomplicated, this offers aromas of honey and sliced peaches, with a lively texture and good balance. Medium-bodied, it makes a great wine … |
| 5 | Domaine des Baumard 2009 Carte d'Or Chenin Blanc (Coteaux du Layon) | Chenin Blanc | France | – | 92 | A wine that contrasts extreme sweetness with freshness. It has beautiful acidity as well as a great swathe of sweet peach and orange laced with … |

**semantic_bge**

| # | Wine | Variety | Country | Price | Points | Tasting note (start) |
|---|---|---|---|---|---|---|
| 1 | Arger-Martucci 2004 Dulcinea Late Harvest Sauvignon Blanc-Semillon (California) | Sauvignon Blanc-Semillon | US | $45 | 90 | This extremely sweet wine tastes of rich, seductive honey. The underlying fruit suggests extremely ripe apricots, peaches, pineapples and vanilla. It's all balanced with fine … |
| 2 | Jana 2007 Old Vine Riesling (Napa Valley) | Riesling | US | $20 | 85 | You could classify this as an off-dry wine, but it's just sweet and honeyed enough to be a dessert wine. With pleasant acidity providing balance, … |
| 3 | Cave de Ribeauvillé 2007 Gloeckelberg Grand Cru Pinot Gris (Alsace) | Pinot Gris | France | – | 90 | This rich, sweet wine is honeyed, with a crisp, balancing lemon character. It is full and rich, touched by spice as well as mango and … |
| 4 | Domaine Cauhapé 2013 Sève d'Automne  (Jurançon Sec) | Gros and Petit Manseng | France | $35 | 92 | This powerful and rich wine wants to be sweet but is bone dry. Hints of honey accompany opulent ripe peach and spiced pear flavors. The … |
| 5 | Barton & Guestier 2008  Sauternes | Bordeaux-style White Blend | France | $25 | 85 | Ripe botrytis, honey and peaches make for an attractive, open, ready-to-drink sweet wine. The acidity keeps the richness in check. |

## Query 5: "A strong, rich red with chocolate and spice for a cold winter night"


**popularity**

| # | Wine | Variety | Country | Price | Points | Tasting note (start) |
|---|---|---|---|---|---|---|
| 1 | Wines & Winemakers 2009 Aguia Moura em Vinhas Velhas Reserva Red (Douro) | Portuguese Red | Portugal | $18 | 93 | This is a smooth, impressively rich wine powered by silky tannins and ripe black fruits. The palate is intensely concentrated with a big, bold structure … |
| 2 | Wines & Winemakers 2012 Companhia das Lezírias Tyto Alba Vinhas Protegidas Touriga Nacional (Tejo) | Touriga Nacional | Portugal | $36 | 93 | Named after the barn owl (Tyto alba) that nests in the woods of the Lezirias estate, this has great tannins that give shape to the … |
| 3 | Wines & Winemakers 2014 Companhia das Lezirias Tyto Alba Touriga Nacional (Tejo) | Touriga Nacional | Portugal | $36 | 93 | This 100% Touriga Nacional comes from the government-owned estate in the north of Tejo. Named after the barn owl that is found in the woods, … |
| 4 | Wines & Winemakers 2015 Companhia das Lezirias 1836 Grande Reserva Branco Fernão Pires (Tejo) | Fernão Pires | Portugal | $77 | 93 | This fine low-production wood-aged wine is almost Burgundian in its creamy richness. The toast is nicely judged giving a wine that has just enough acidity … |
| 5 | Wines & Winemakers 2012 Pegos Claros Vinhas Velhas Reserva Castelão (Palmela) | Castelão | Portugal | $18 | 93 | A serious and dense wine, this is rich and full of dark juicy fruit. It has plenty of structure of course, with its rich tannins … |

**tfidf**

| # | Wine | Variety | Country | Price | Points | Tasting note (start) |
|---|---|---|---|---|---|---|
| 1 | Renieri 2008  Brunello di Montalcino | Sangiovese Grosso | Italy | $50 | 91 | Renieri delivers that style of austere and elegant Brunello that begs for a steak dinner on a cold winter night. The wine is fleshy and … |
| 2 | Casato1 2007 Voliero  (Brunello di Montalcino) | Sangiovese Grosso | Italy | $50 | 90 | An inky-black appearance and spicy-savory aromas make this the kind of Brunello you want to drink on a really cold winter night. Tightly concentrated and … |
| 3 | Donna Olga 2008  Brunello di Montalcino | Sangiovese Grosso | Italy | – | 90 | This is the kind of Brunello you want on a cold winter night by the fireplace. The wine delivers dark concentration with oak spice and … |
| 4 | Henry Fessy 2010  Beaujolais-Villages | Gamay | France | $12 | 85 | A series of firm tannins goes with acidity, raspberry fruits and attractive juicy sweetness. A wine for drinking on cold winter evenings. |
| 5 | Midlife Crisis 2006 Barbera (Paso Robles) | Barbera | US | $21 | 86 | Awfully high in alcohol, but it wears it well, with fat, glyceriney flavors of blackberry and cherry jam, chocolate, anise, red currants and sweet oak. … |

**attribute**

| # | Wine | Variety | Country | Price | Points | Tasting note (start) |
|---|---|---|---|---|---|---|
| 1 | Tenuta dell'Ornellaia 2007 Masseto Merlot (Toscana) | Merlot | Italy | $460 | 100 | A perfect wine from a classic vintage, the 2007 Masseto (100% Merlot from a 17-acre vineyard of the same name) opens with an unabashedly opulent … |
| 2 | Casa Ferreirinha 2008 Barca-Velha Red (Douro) | Portuguese Red | Portugal | $450 | 100 | This is the latest release of what has long been regarded as Portugal's iconic wine. And it is magnificent. The last vintage was the 2004 … |
| 3 | Biondi Santi 2010 Riserva  (Brunello di Montalcino) | Sangiovese | Italy | $550 | 100 | This gorgeous, fragrant wine opens with classic Sangiovese scents of violet, rose, perfumed red berry, new leather and a whiff of baking spice. The elegant, … |
| 4 | Cardinale 2006 Cabernet Sauvignon (Napa Valley) | Cabernet Sauvignon | US | $200 | 100 | Tasted in a flight of great and famous Napa wines, this Cardinale stood at the head of the pack. Starts with a very fine nose … |
| 5 | Château Léoville Barton 2010  Saint-Julien | Bordeaux-style Red Blend | France | $150 | 100 | This is a magnificently solid wine, initially even a bit severe. At this young stage, the tannins dominate, but it's also full of black fruit … |

**semantic**

| # | Wine | Variety | Country | Price | Points | Tasting note (start) |
|---|---|---|---|---|---|---|
| 1 | Golser Wein 2014 St. Laurent (Burgenland) | St. Laurent | Austria | $22 | 89 | Smoky notes and a slight hint of chocolate spice this very plummy yet fresh, light-bodied red. The tannins are slight but beautifully present and there … |
| 2 | Terra Romana 2006 Cuvée Charlotte Red (Dealu Mare) | Red Blend | Romania | $17 | 84 | This intriguing red begins with cinnamon, leather and pepper aromas, followed by a nice balance of warming spices and rich red fruit. Good complexity with … |
| 3 | Gerovassiliou 2006 Syrah (Epanomi) | Syrah | Greece | – | 88 | This full-bodied red gives sexy aromas of dark chocolate, cherry, vanilla and cedar, followed by equally deep flavors of cocoa, black fruit, plum and black … |
| 4 | Woop Woop 2009 The Chook Shiraz-Viognier (South Australia) | Shiraz-Viognier | Australia | $19 | 89 | A charmingly complex medium-weight red, with raspberry fruit, peppery spice and dark chocolate notes all coated in a soft, fine dusting of tannins. Drink now–2014. |
| 5 | Kirnbauer 2011 Vitikult Riede Neuberg Blaufränkisch (Mittelburgenland) | Blaufränkisch | Austria | $16 | 86 | Here's a slender but aromatic red, with a pleasing cherry-flesh note and a cedar spice accent. It's fresh, sprightly and very fruity. |

**semantic_bge**

| # | Wine | Variety | Country | Price | Points | Tasting note (start) |
|---|---|---|---|---|---|---|
| 1 | Cellier des Chartreux 2011 Domaine des Colombettes Les Terres du Roy Red (Lirac) | Rhône-style Red Blend | France | $24 | 87 | This is a rich, weighty red with a velvety texture and plenty of alcoholic warmth. The fruit is ripe to the point of turning chocolaty … |
| 2 | Gerovassiliou 2006 Syrah (Epanomi) | Syrah | Greece | – | 88 | This full-bodied red gives sexy aromas of dark chocolate, cherry, vanilla and cedar, followed by equally deep flavors of cocoa, black fruit, plum and black … |
| 3 | Palivou 2004 Ammos Terra Leone Single Vineyard Selection Agiorgitiko (Nemea) | Agiorgitiko | Greece | $45 | 87 | A darker, denser red that offers chocolate, black fruit and spice on the nose and palate. A spin of nutty spice and dried fruit punctuates … |
| 4 | B Cellars 2014 Ehrlich Vineyard Cabernet Sauvignon (Oakville) | Cabernet Sauvignon | US | $118 | 92 | Smoky and oaky with a punch of dark chocolate, this full-bodied, brambly red bursts with clove, cinnamon and blackberry flavors. It develops further layers as … |
| 5 | Stemilt Creek 2004 Merlot (Washington) | Merlot | US | $20 | 89 | This delicious wine is as soft, silky and supple as can be. The fruit is slathered in spicy, chocolaty oak, which lends some sharpness to … |
