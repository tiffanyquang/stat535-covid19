-- MySQL dump 10.13  Distrib 8.0.19, for Win64 (x86_64)
--
-- Host: 127.0.0.1    Database: stats_535_data_base
-- ------------------------------------------------------
-- Server version	8.0.19

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `master_state_data`
--

DROP TABLE IF EXISTS `master_state_data`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `master_state_data` (
  `STATE` text,
  `abbr` text,
  `Density` double DEFAULT NULL,
  `Pop` int DEFAULT NULL,
  `LandArea` int DEFAULT NULL,
  `gdpRank` int DEFAULT NULL,
  `stateGDP` int DEFAULT NULL,
  `stateGDPperc` double DEFAULT NULL,
  `gdpGrowth2018` double DEFAULT NULL,
  `Airports` int DEFAULT NULL,
  `Automobiles` double DEFAULT NULL,
  `Buses` int DEFAULT NULL,
  `pvi` int DEFAULT NULL,
  `governorParty` text,
  `senateParty` text,
  `houseBalance` text,
  `positive` int DEFAULT NULL,
  `positiveScore` int DEFAULT NULL,
  `negativeScore` int DEFAULT NULL,
  `negativeRegularScore` int DEFAULT NULL,
  `commercialScore` int DEFAULT NULL,
  `grade` text,
  `score` int DEFAULT NULL,
  `negative` int DEFAULT NULL,
  `pending` text,
  `hospitalized` text,
  `death` int DEFAULT NULL,
  `total` int DEFAULT NULL,
  `totalTestResults` int DEFAULT NULL,
  `fips` int DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `master_state_data`
--

LOCK TABLES `master_state_data` WRITE;
/*!40000 ALTER TABLE `master_state_data` DISABLE KEYS */;
INSERT INTO `master_state_data` VALUES ('Alaska','AK',1.2863,734002,570641,46,54851,0.003,-0.3,767,173487,8080,9,'Republican','Republican','1R',133,1,1,1,1,'A',4,4470,'','9',3,4603,4603,2),('Alabama','AL',96.9221,4908621,50645,27,224654,0.011,2,316,2065463,5682,14,'Republican','Both','6R, 1D',1077,1,1,0,1,'B',3,6697,'','',26,7774,7774,1),('Arkansas','AR',58.403,3038999,52035,34,129812,0.006,0.9,304,926789,11941,15,'Republican','Republican','4R',584,1,1,1,1,'A',4,7354,'','90',10,7938,7938,5),('Arizona','AZ',64.955,7378494,113594,20,355311,0.017,4,298,2391624,9051,5,'Republican','Both','5D, 4R',1413,1,1,0,1,'B',3,19645,'','149',29,21058,21058,4),('California','CA',256.3727,39937489,155779,1,3018337,0.145,3.5,882,14860967,99917,-12,'Democratic','Democratic','46D, 7R',8155,1,1,1,0,'B',3,21772,'57400','1855',171,87327,29927,6),('Colorado','CO',56.4011,5845526,103642,16,376994,0.018,3.5,447,1785058,13305,-1,'Democratic','Both','4D, 3R',2966,1,1,1,1,'A',4,13883,'','509',69,16849,16849,8),('Connecticut','CT',735.8689,3563077,4842,23,279653,0.013,1,123,1302339,11378,-6,'Democratic','Democratic','5D',3557,1,1,1,1,'A',4,13043,'','766',85,16600,16600,9),('District Of Columbia','DC',11814.541,720687,61,NULL,NULL,NULL,NULL,16,208141,4885,NULL,NULL,NULL,NULL,586,1,1,1,1,'A',4,3262,'2','',11,3850,3848,11),('Delaware','DE',504.3073,982895,1949,42,76537,0.004,0.3,42,419199,3763,-6,'Democratic','Democratic','1D',368,1,1,1,1,'A',4,4015,'','57',11,4383,4383,10),('Florida','FL',410.1256,21992985,53625,4,1059144,0.051,3.5,875,7778493,59596,2,'Republican','Republican','14R, 13D',6955,1,1,1,1,'A',4,59529,'1235','949',87,67719,66484,12),('Georgia','GA',186.6719,10736059,57513,9,601503,0.029,2.6,454,3536055,36418,5,'Republican','Republican','9R, 5D',4638,1,1,1,1,'A',4,15688,'','952',139,20326,20326,13),('Hawaii','HI',219.9419,1412687,6423,38,93419,0.004,1,46,512386,3033,-18,'Democratic','Democratic','2D',208,1,1,1,1,'A',4,8721,'18','13',1,8947,8929,15),('Iowa','IA',56.9284,3179849,55857,30,192608,0.009,1.4,279,1268996,9284,3,'Republican','Republican','3D, 1R',549,1,1,1,1,'A',4,7304,'','99',9,7853,7853,19),('Idaho','ID',22.0969,1826156,82643,41,78640,0.004,4.1,301,550006,3870,19,'Republican','Republican','2R',525,1,1,1,1,'A',4,6076,'','46',9,6601,6601,16),('Illinois','IL',228.0243,12659682,55519,5,879947,0.042,2.1,707,4655844,34359,-7,'Democratic','Democratic','13D, 5R',6980,1,1,1,1,'A',4,33404,'','',141,40384,40384,17),('Indiana','IN',188.281,6745354,35826,19,371629,0.018,1.9,518,2261440,20725,9,'Republican','Republican','7R, 2D',2565,1,1,1,0,'B',3,11810,'','',65,14375,14375,18),('Kansas','KS',35.5968,2910357,81759,32,169558,0.008,1.9,372,987259,6228,13,'Democratic','Republican','3R, 1D',482,1,1,1,0,'B',3,5411,'','114',10,5893,5893,20),('Kentucky','KY',113.9566,4499692,39486,28,211621,0.01,1.4,263,1691085,10759,15,'Republican','Republican','5R, 1D',591,1,1,1,1,'A',4,6965,'','',17,7556,7556,21),('Louisiana','LA',107.5175,4645184,43204,24,255492,0.012,1.1,498,1389436,29160,11,'Democratic','Republican','5R, 1D',6424,1,1,1,1,'A',4,39352,'','1498',273,45776,45776,22),('Massachusetts','MA',894.4355,6976597,7800,11,575635,0.028,2.7,225,2208468,13498,-12,'Republican','Democratic','9D',7738,1,1,0,1,'B',3,44000,'','682',122,51738,51738,25),('Maryland','MD',626.6731,6083116,9707,15,417776,0.02,1.6,223,2006904,22972,-12,'Republican','Democratic','7D, 1R',1985,1,1,1,1,'A',4,17233,'','522',31,19218,19218,24),('Maine','ME',43.6336,1345790,30843,43,65349,0.003,1.9,217,387233,4339,-3,'Democratic','Both*','2D',344,1,1,1,1,'A',4,6088,'','63',7,6432,6432,23),('Michigan','MI',177.6655,10045029,56539,14,537087,0.026,2.7,481,3257473,8737,-1,'Democratic','Democratic','7D, 6R, 1I',9334,1,1,0,1,'B',3,11893,'','',337,21227,21227,26),('Minnesota','MN',71.5922,5700671,79627,17,374920,0.018,2.2,475,2146187,19980,-1,'Democratic','Democratic','5D, 3R',689,1,1,1,1,'A',4,20502,'','122',17,21191,21191,27),('Missouri','MO',89.7453,6169270,68742,22,323287,0.015,2.3,510,2141050,27180,9,'Republican','Republican','6R, 2D',1581,1,1,1,1,'A',4,15846,'','',18,17427,17427,29),('Mississippi','MS',63.7056,2989260,46923,36,115749,0.006,1,235,822864,8018,9,'Republican','Republican','3R, 1D',1073,1,1,1,0,'B',3,3712,'','332',22,4785,4785,28),('Montana','MT',7.4668,1086759,145546,48,49635,0.002,0.9,301,441392,4702,11,'Democratic','Both','1R',208,1,1,1,1,'A',4,4710,'','17',5,4918,4918,30),('North Carolina','NC',218.2702,10611862,48618,12,575605,0.028,2.9,486,3317918,32493,3,'Democratic','Republican','8R, 3D, 2 Vacant',1584,1,1,1,1,'A',4,24659,'','204',10,26243,26243,37),('North Dakota','ND',11.0393,761723,69001,45,55657,0.003,2.5,282,276545,3331,17,'Republican','Republican','1R',142,1,1,1,0,'B',3,4351,'','23',3,4493,4493,38),('Nebraska','NE',25.4161,1952570,76824,35,124742,0.006,1.5,246,677040,13564,14,'Republican','Republican','3R',210,1,1,1,1,'A',4,3475,'8','',4,3693,3685,31),('New Hampshire','NH',153.1605,1371246,8953,39,86046,0.004,2.2,186,505381,2871,0,'Republican','Democratic','2D',415,1,1,1,0,'B',3,5985,'97','59',4,6497,6400,33),('New Jersey','NJ',1215.1991,8936574,7354,8,634721,0.03,2,320,2798699,25097,-7,'Democratic','Democratic','11D, 1R',22255,1,1,1,1,'A',4,30387,'','',355,52642,52642,34),('New Mexico','NM',17.285,2096640,121298,37,101452,0.005,1.8,171,609494,6278,-3,'Democratic','Democratic','3D',315,1,1,1,1,'A',4,12925,'','24',5,13240,13240,35),('Nevada','NV',28.5993,3139658,109781,33,168752,0.008,3.2,139,1047508,3198,-1,'Democratic','Democratic','3D, 1R',1279,1,1,1,1,'A',4,11519,'','',26,12798,12798,32),('New York','NY',412.5211,19440469,47126,3,1701399,0.082,2.1,551,4563035,80446,-12,'Democratic','Democratic','21D, 6R',83712,1,1,1,1,'A',4,137168,'','18368',1941,220880,220880,36),('Ohio','OH',287.5038,11747694,40861,7,689139,0.033,1.8,660,4605459,41255,3,'Republican','Both','12R, 4D',2547,1,1,0,1,'B',3,26992,'','679',65,29539,29539,39),('Oklahoma','OK',57.6547,3954821,68595,29,203250,0.01,1.8,457,1337624,3155,20,'Republican','Republican','4R, 1D',719,1,1,1,0,'B',3,1248,'','219',30,1967,1967,40),('Oregon','OR',44.8086,4301089,95988,25,243085,0.012,3.4,442,1542461,17691,-5,'Democratic','Democratic','4D, 1R',690,1,1,1,1,'A',4,13136,'','154',18,13826,13826,41),('Pennsylvania','PA',286.5449,12820878,44743,6,803307,0.038,2.1,758,4463267,55935,0,'Democratic','Both','9D, 9R',5805,1,1,1,1,'A',4,42427,'','620',74,48232,48232,42),('Rhode Island','RI',1021.4323,1056161,1034,44,61341,0.003,0.6,22,420313,2273,-10,'Democratic','Democratic','2D',566,1,1,1,0,'B',3,3831,'','60',10,4397,4397,44),('South Carolina','SC',173.3174,5210095,30061,26,234367,0.011,1.6,196,1808401,17728,8,'Republican','Republican','5R, 2D',1293,1,1,1,0,'B',3,5033,'','349',26,6326,6326,45),('South Dakota','SD',11.9116,903027,75811,47,52544,0.003,1.3,185,353180,2614,14,'Republican','Republican','1R',129,1,1,1,1,'A',4,3903,'0','12',2,4032,4032,46),('Tennessee','TN',167.2748,6897576,41235,18,373663,0.018,3,339,2325004,29158,14,'Republican','Republican','7R, 2D',2683,1,1,1,1,'A',4,29769,'','200',24,32452,32452,47),('Texas','TX',112.8204,29472295,261232,2,1818585,0.087,3.2,2031,8176879,67143,8,'Republican','Republican','23R, 13D',3997,1,1,1,1,'A',4,43860,'','196',58,47857,47857,48),('Utah','UT',39.943,3282115,82170,31,180862,0.009,4.3,156,925052,6109,20,'Republican','Republican','3R, 1D',1012,1,1,1,1,'A',4,20155,'','91',7,21167,21167,49),('Virginia','VA',218.4403,8626207,39490,13,544348,0.026,2.8,431,3251339,34268,-1,'Democratic','Democratic','7D, 4R',1484,1,1,1,1,'A',4,13860,'','305',34,15344,15344,51),('Vermont','VT',68.1416,628061,9217,50,34154,0.002,1.2,89,221936,1070,-15,'Republican','Democratic*','1D',321,1,1,1,1,'A',4,4174,'','45',16,4495,4495,50),('Washington','WA',117.3272,7797095,66456,10,576624,0.028,5.7,538,2925765,23566,-7,'Democratic','Democratic','7D, 3R',5634,1,1,1,1,'A',4,60566,'','254',224,66200,66200,53),('Wisconsin','WI',108.0497,5851754,54158,21,342470,0.016,2.5,540,2092584,15567,0,'Democratic','Both','5R, 3D',1550,1,1,1,1,'A',4,18819,'','398',24,20369,20369,55),('West Virginia','WV',73.9691,1778070,24038,40,79168,0.004,2.4,125,562984,3342,19,'Republican','Both','3R',191,1,1,1,0,'B',3,4384,'','1',1,4575,4575,54),('Wyoming','WY',5.84,567025,97093,49,39899,0.002,0.3,121,193523,4219,25,'Republican','Republican','1R',130,1,1,1,1,'A',4,2218,'','18',0,2348,2348,56);
/*!40000 ALTER TABLE `master_state_data` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `name_abbr`
--

DROP TABLE IF EXISTS `name_abbr`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `name_abbr` (
  `STATE` text,
  `abbr` text
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `name_abbr`
--

LOCK TABLES `name_abbr` WRITE;
/*!40000 ALTER TABLE `name_abbr` DISABLE KEYS */;
INSERT INTO `name_abbr` VALUES ('Alabama','AL'),('Alaska','AK'),('American Samoa','AS'),('Arizona','AZ'),('Arkansas','AR'),('California','CA'),('Colorado','CO'),('Connecticut','CT'),('Delaware','DE'),('District Of Columbia','DC'),('Federated States Of Micronesia','FM'),('Florida','FL'),('Georgia','GA'),('Guam','GU'),('Hawaii','HI'),('Idaho','ID'),('Illinois','IL'),('Indiana','IN'),('Iowa','IA'),('Kansas','KS'),('Kentucky','KY'),('Louisiana','LA'),('Maine','ME'),('Marshall Islands','MH'),('Maryland','MD'),('Massachusetts','MA'),('Michigan','MI'),('Minnesota','MN'),('Mississippi','MS'),('Missouri','MO'),('Montana','MT'),('Nebraska','NE'),('Nevada','NV'),('New Hampshire','NH'),('New Jersey','NJ'),('New Mexico','NM'),('New York','NY'),('North Carolina','NC'),('North Dakota','ND'),('Northern Mariana Islands','MP'),('Ohio','OH'),('Oklahoma','OK'),('Oregon','OR'),('Palau','PW'),('Pennsylvania','PA'),('Puerto Rico','PR'),('Rhode Island','RI'),('South Carolina','SC'),('South Dakota','SD'),('Tennessee','TN'),('Texas','TX'),('Utah','UT'),('Vermont','VT'),('Virgin Islands','VI'),('Virginia','VA'),('Washington','WA'),('West Virginia','WV'),('Wisconsin','WI'),('Wyoming','WY');
/*!40000 ALTER TABLE `name_abbr` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `state_density`
--

DROP TABLE IF EXISTS `state_density`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `state_density` (
  `State` text,
  `Density` double DEFAULT NULL,
  `Pop` int DEFAULT NULL,
  `LandArea` int DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `state_density`
--

LOCK TABLES `state_density` WRITE;
/*!40000 ALTER TABLE `state_density` DISABLE KEYS */;
INSERT INTO `state_density` VALUES ('District of Columbia',11814.541,720687,61),('New Jersey',1215.1991,8936574,7354),('Rhode Island',1021.4323,1056161,1034),('Massachusetts',894.4355,6976597,7800),('Connecticut',735.8689,3563077,4842),('Maryland',626.6731,6083116,9707),('Delaware',504.3073,982895,1949),('New York',412.5211,19440469,47126),('Florida',410.1256,21992985,53625),('Ohio',287.5038,11747694,40861),('Pennsylvania',286.5449,12820878,44743),('California',256.3727,39937489,155779),('Illinois',228.0243,12659682,55519),('Hawaii',219.9419,1412687,6423),('Virginia',218.4403,8626207,39490),('North Carolina',218.2702,10611862,48618),('Indiana',188.281,6745354,35826),('Georgia',186.6719,10736059,57513),('Michigan',177.6655,10045029,56539),('South Carolina',173.3174,5210095,30061),('Tennessee',167.2748,6897576,41235),('New Hampshire',153.1605,1371246,8953),('Washington',117.3272,7797095,66456),('Kentucky',113.9566,4499692,39486),('Texas',112.8204,29472295,261232),('Wisconsin',108.0497,5851754,54158),('Louisiana',107.5175,4645184,43204),('Alabama',96.9221,4908621,50645),('Missouri',89.7453,6169270,68742),('West Virginia',73.9691,1778070,24038),('Minnesota',71.5922,5700671,79627),('Vermont',68.1416,628061,9217),('Arizona',64.955,7378494,113594),('Mississippi',63.7056,2989260,46923),('Arkansas',58.403,3038999,52035),('Oklahoma',57.6547,3954821,68595),('Iowa',56.9284,3179849,55857),('Colorado',56.4011,5845526,103642),('Oregon',44.8086,4301089,95988),('Maine',43.6336,1345790,30843),('Utah',39.943,3282115,82170),('Kansas',35.5968,2910357,81759),('Nevada',28.5993,3139658,109781),('Nebraska',25.4161,1952570,76824),('Idaho',22.0969,1826156,82643),('New Mexico',17.285,2096640,121298),('South Dakota',11.9116,903027,75811),('North Dakota',11.0393,761723,69001),('Montana',7.4668,1086759,145546),('Wyoming',5.84,567025,97093),('Alaska',1.2863,734002,570641);
/*!40000 ALTER TABLE `state_density` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `state_gdp`
--

DROP TABLE IF EXISTS `state_gdp`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `state_gdp` (
  `State` text,
  `gdpRank` int DEFAULT NULL,
  `stateGDP` int DEFAULT NULL,
  `stateGDPperc` double DEFAULT NULL,
  `gdpGrowth2018` double DEFAULT NULL,
  `Pop` int DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `state_gdp`
--

LOCK TABLES `state_gdp` WRITE;
/*!40000 ALTER TABLE `state_gdp` DISABLE KEYS */;
INSERT INTO `state_gdp` VALUES ('California',1,3018337,0.145,3.5,39937489),('Texas',2,1818585,0.087,3.2,29472295),('New York',3,1701399,0.082,2.1,19440469),('Florida',4,1059144,0.051,3.5,21992985),('Illinois',5,879947,0.042,2.1,12659682),('Pennsylvania',6,803307,0.038,2.1,12820878),('Ohio',7,689139,0.033,1.8,11747694),('New Jersey',8,634721,0.03,2,8936574),('Georgia',9,601503,0.029,2.6,10736059),('Washington',10,576624,0.028,5.7,7797095),('Massachusetts',11,575635,0.028,2.7,6976597),('North Carolina',12,575605,0.028,2.9,10611862),('Virginia',13,544348,0.026,2.8,8626207),('Michigan',14,537087,0.026,2.7,10045029),('Maryland',15,417776,0.02,1.6,6083116),('Colorado',16,376994,0.018,3.5,5845526),('Minnesota',17,374920,0.018,2.2,5700671),('Tennessee',18,373663,0.018,3,6897576),('Indiana',19,371629,0.018,1.9,6745354),('Arizona',20,355311,0.017,4,7378494),('Wisconsin',21,342470,0.016,2.5,5851754),('Missouri',22,323287,0.015,2.3,6169270),('Connecticut',23,279653,0.013,1,3563077),('Louisiana',24,255492,0.012,1.1,4645184),('Oregon',25,243085,0.012,3.4,4301089),('South Carolina',26,234367,0.011,1.6,5210095),('Alabama',27,224654,0.011,2,4908621),('Kentucky',28,211621,0.01,1.4,4499692),('Oklahoma',29,203250,0.01,1.8,3954821),('Iowa',30,192608,0.009,1.4,3179849),('Utah',31,180862,0.009,4.3,3282115),('Kansas',32,169558,0.008,1.9,2910357),('Nevada',33,168752,0.008,3.2,3139658),('Arkansas',34,129812,0.006,0.9,3038999),('Nebraska',35,124742,0.006,1.5,1952570),('Mississippi',36,115749,0.006,1,2989260),('New Mexico',37,101452,0.005,1.8,2096640),('Hawaii',38,93419,0.004,1,1412687),('New Hampshire',39,86046,0.004,2.2,1371246),('West Virginia',40,79168,0.004,2.4,1778070),('Idaho',41,78640,0.004,4.1,1826156),('Delaware',42,76537,0.004,0.3,982895),('Maine',43,65349,0.003,1.9,1345790),('Rhode Island',44,61341,0.003,0.6,1056161),('North Dakota',45,55657,0.003,2.5,761723),('Alaska',46,54851,0.003,-0.3,734002),('South Dakota',47,52544,0.003,1.3,903027),('Montana',48,49635,0.002,0.9,1086759),('Wyoming',49,39899,0.002,0.3,567025),('Vermont',50,34154,0.002,1.2,628061);
/*!40000 ALTER TABLE `state_gdp` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `state_transport`
--

DROP TABLE IF EXISTS `state_transport`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `state_transport` (
  `STATE` text,
  `Airports` text,
  `Automobiles` text,
  `Buses` text
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `state_transport`
--

LOCK TABLES `state_transport` WRITE;
/*!40000 ALTER TABLE `state_transport` DISABLE KEYS */;
INSERT INTO `state_transport` VALUES ('Alabama','316','2,065,463','5,682'),('Alaska','767','173,487','8,080'),('American Samoa','','',''),('Arizona','298','2,391,624','9,051'),('Arkansas','304','926,789','11,941'),('California','882','14,860,967','99,917'),('Colorado','447','1,785,058','13,305'),('Connecticut','123','1,302,339','11,378'),('Delaware','42','419,199','3,763'),('District Of Columbia','16','208,141','4,885'),('Federated States Of Micronesia','','',''),('Florida','875','7,778,493','59,596'),('Georgia','454','3,536,055','36,418'),('Guam','4','',''),('Hawaii','46','512,386','3,033'),('Idaho','301','550,006','3,870'),('Illinois','707','4,655,844','34,359'),('Indiana','518','2,261,440','20,725'),('Iowa','279','1,268,996','9,284'),('Kansas','372','987,259','6,228'),('Kentucky','263','1,691,085','10,759'),('Louisiana','498','1,389,436','29,160'),('Maine','217','387,233','4,339'),('Marshall Islands','','',''),('Maryland','223','2,006,904','22,972'),('Massachusetts','225','2,208,468','13,498'),('Michigan','481','3,257,473','8,737'),('Minnesota','475','2,146,187','19,980'),('Mississippi','235','822,864','8,018'),('Missouri','510','2,141,050','27,180'),('Montana','301','441,392','4,702'),('Nebraska','246','677,040','13,564'),('Nevada','139','1,047,508','3,198'),('New Hampshire','186','505,381','2,871'),('New Jersey','320','2,798,699','25,097'),('New Mexico','171','609,494','6,278'),('New York','551','4,563,035','80,446'),('North Carolina','486','3,317,918','32,493'),('North Dakota','282','276,545','3,331'),('Northern Mariana Islands','','',''),('Ohio','660','4,605,459','41,255'),('Oklahoma','457','1,337,624','3,155'),('Oregon','442','1,542,461','17,691'),('Palau','','',''),('Pennsylvania','758','4,463,267','55,935'),('Puerto Rico','47','',''),('Rhode Island','22','420,313','2,273'),('South Carolina','196','1,808,401','17,728'),('South Dakota','185','353,180','2,614'),('Tennessee','339','2,325,004','29,158'),('Texas','2031','8,176,879','67,143'),('Utah','156','925,052','6,109'),('Vermont','89','221,936','1,070'),('Virgin Islands','','',''),('Virginia','431','3,251,339','34,268'),('Washington','538','2,925,765','23,566'),('West Virginia','125','562,984','3,342'),('Wisconsin','540','2,092,584','15,567'),('Wyoming','121','193,523','4,219');
/*!40000 ALTER TABLE `state_transport` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `states_by_color`
--

DROP TABLE IF EXISTS `states_by_color`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `states_by_color` (
  `State` text,
  `pvi` int DEFAULT NULL,
  `governorParty` text,
  `senateParty` text,
  `houseBalance` text,
  `Pop` int DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `states_by_color`
--

LOCK TABLES `states_by_color` WRITE;
/*!40000 ALTER TABLE `states_by_color` DISABLE KEYS */;
INSERT INTO `states_by_color` VALUES ('Hawaii',-18,'Democratic','Democratic','2D',1412687),('Vermont',-15,'Republican','Democratic*','1D',628061),('California',-12,'Democratic','Democratic','46D, 7R',39937489),('Maryland',-12,'Republican','Democratic','7D, 1R',6083116),('Massachusetts',-12,'Republican','Democratic','9D',6976597),('New York',-12,'Democratic','Democratic','21D, 6R',19440469),('Rhode Island',-10,'Democratic','Democratic','2D',1056161),('Illinois',-7,'Democratic','Democratic','13D, 5R',12659682),('New Jersey',-7,'Democratic','Democratic','11D, 1R',8936574),('Washington',-7,'Democratic','Democratic','7D, 3R',7797095),('Connecticut',-6,'Democratic','Democratic','5D',3563077),('Delaware',-6,'Democratic','Democratic','1D',982895),('Oregon',-5,'Democratic','Democratic','4D, 1R',4301089),('Maine',-3,'Democratic','Both*','2D',1345790),('New Mexico',-3,'Democratic','Democratic','3D',2096640),('Colorado',-1,'Democratic','Both','4D, 3R',5845526),('Michigan',-1,'Democratic','Democratic','7D, 6R, 1I',10045029),('Minnesota',-1,'Democratic','Democratic','5D, 3R',5700671),('Nevada',-1,'Democratic','Democratic','3D, 1R',3139658),('Virginia',-1,'Democratic','Democratic','7D, 4R',8626207),('New Hampshire',0,'Republican','Democratic','2D',1371246),('Pennsylvania',0,'Democratic','Both','9D, 9R',12820878),('Wisconsin',0,'Democratic','Both','5R, 3D',5851754),('Florida',2,'Republican','Republican','14R, 13D',21992985),('Iowa',3,'Republican','Republican','3D, 1R',3179849),('North Carolina',3,'Democratic','Republican','8R, 3D, 2 Vacant',10611862),('Ohio',3,'Republican','Both','12R, 4D',11747694),('Arizona',5,'Republican','Both','5D, 4R',7378494),('Georgia',5,'Republican','Republican','9R, 5D',10736059),('South Carolina',8,'Republican','Republican','5R, 2D',5210095),('Texas',8,'Republican','Republican','23R, 13D',29472295),('Alaska',9,'Republican','Republican','1R',734002),('Indiana',9,'Republican','Republican','7R, 2D',6745354),('Mississippi',9,'Republican','Republican','3R, 1D',2989260),('Missouri',9,'Republican','Republican','6R, 2D',6169270),('Louisiana',11,'Democratic','Republican','5R, 1D',4645184),('Montana',11,'Democratic','Both','1R',1086759),('Kansas',13,'Democratic','Republican','3R, 1D',2910357),('Alabama',14,'Republican','Both','6R, 1D',4908621),('Nebraska',14,'Republican','Republican','3R',1952570),('South Dakota',14,'Republican','Republican','1R',903027),('Tennessee',14,'Republican','Republican','7R, 2D',6897576),('Arkansas',15,'Republican','Republican','4R',3038999),('Kentucky',15,'Republican','Republican','5R, 1D',4499692),('North Dakota',17,'Republican','Republican','1R',761723),('Idaho',19,'Republican','Republican','2R',1826156),('West Virginia',19,'Republican','Both','3R',1778070),('Oklahoma',20,'Republican','Republican','4R, 1D',3954821),('Utah',20,'Republican','Republican','3R, 1D',3282115),('Wyoming',25,'Republican','Republican','1R',567025);
/*!40000 ALTER TABLE `states_by_color` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `states_covidcases`
--

DROP TABLE IF EXISTS `states_covidcases`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `states_covidcases` (
  `state_abbr` text,
  `positive` int DEFAULT NULL,
  `positiveScore` int DEFAULT NULL,
  `negativeScore` int DEFAULT NULL,
  `negativeRegularScore` int DEFAULT NULL,
  `commercialScore` int DEFAULT NULL,
  `grade` text,
  `score` int DEFAULT NULL,
  `negative` int DEFAULT NULL,
  `pending` text,
  `hospitalized` text,
  `death` int DEFAULT NULL,
  `total` int DEFAULT NULL,
  `totalTestResults` int DEFAULT NULL,
  `fips` int DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `states_covidcases`
--

LOCK TABLES `states_covidcases` WRITE;
/*!40000 ALTER TABLE `states_covidcases` DISABLE KEYS */;
INSERT INTO `states_covidcases` VALUES ('AK',133,1,1,1,1,'A',4,4470,'','9',3,4603,4603,2),('AL',1077,1,1,0,1,'B',3,6697,'','',26,7774,7774,1),('AR',584,1,1,1,1,'A',4,7354,'','90',10,7938,7938,5),('AZ',1413,1,1,0,1,'B',3,19645,'','149',29,21058,21058,4),('CA',8155,1,1,1,0,'B',3,21772,'57400','1855',171,87327,29927,6),('CO',2966,1,1,1,1,'A',4,13883,'','509',69,16849,16849,8),('CT',3557,1,1,1,1,'A',4,13043,'','766',85,16600,16600,9),('DC',586,1,1,1,1,'A',4,3262,'2','',11,3850,3848,11),('DE',368,1,1,1,1,'A',4,4015,'','57',11,4383,4383,10),('FL',6955,1,1,1,1,'A',4,59529,'1235','949',87,67719,66484,12),('GA',4638,1,1,1,1,'A',4,15688,'','952',139,20326,20326,13),('HI',208,1,1,1,1,'A',4,8721,'18','13',1,8947,8929,15),('IA',549,1,1,1,1,'A',4,7304,'','99',9,7853,7853,19),('ID',525,1,1,1,1,'A',4,6076,'','46',9,6601,6601,16),('IL',6980,1,1,1,1,'A',4,33404,'','',141,40384,40384,17),('IN',2565,1,1,1,0,'B',3,11810,'','',65,14375,14375,18),('KS',482,1,1,1,0,'B',3,5411,'','114',10,5893,5893,20),('KY',591,1,1,1,1,'A',4,6965,'','',17,7556,7556,21),('LA',6424,1,1,1,1,'A',4,39352,'','1498',273,45776,45776,22),('MA',7738,1,1,0,1,'B',3,44000,'','682',122,51738,51738,25),('MD',1985,1,1,1,1,'A',4,17233,'','522',31,19218,19218,24),('ME',344,1,1,1,1,'A',4,6088,'','63',7,6432,6432,23),('MI',9334,1,1,0,1,'B',3,11893,'','',337,21227,21227,26),('MN',689,1,1,1,1,'A',4,20502,'','122',17,21191,21191,27),('MO',1581,1,1,1,1,'A',4,15846,'','',18,17427,17427,29),('MS',1073,1,1,1,0,'B',3,3712,'','332',22,4785,4785,28),('MT',208,1,1,1,1,'A',4,4710,'','17',5,4918,4918,30),('NC',1584,1,1,1,1,'A',4,24659,'','204',10,26243,26243,37),('ND',142,1,1,1,0,'B',3,4351,'','23',3,4493,4493,38),('NE',210,1,1,1,1,'A',4,3475,'8','',4,3693,3685,31),('NH',415,1,1,1,0,'B',3,5985,'97','59',4,6497,6400,33),('NJ',22255,1,1,1,1,'A',4,30387,'','',355,52642,52642,34),('NM',315,1,1,1,1,'A',4,12925,'','24',5,13240,13240,35),('NV',1279,1,1,1,1,'A',4,11519,'','',26,12798,12798,32),('NY',83712,1,1,1,1,'A',4,137168,'','18368',1941,220880,220880,36),('OH',2547,1,1,0,1,'B',3,26992,'','679',65,29539,29539,39),('OK',719,1,1,1,0,'B',3,1248,'','219',30,1967,1967,40),('OR',690,1,1,1,1,'A',4,13136,'','154',18,13826,13826,41),('PA',5805,1,1,1,1,'A',4,42427,'','620',74,48232,48232,42),('RI',566,1,1,1,0,'B',3,3831,'','60',10,4397,4397,44),('SC',1293,1,1,1,0,'B',3,5033,'','349',26,6326,6326,45),('SD',129,1,1,1,1,'A',4,3903,'0','12',2,4032,4032,46),('TN',2683,1,1,1,1,'A',4,29769,'','200',24,32452,32452,47),('TX',3997,1,1,1,1,'A',4,43860,'','196',58,47857,47857,48),('UT',1012,1,1,1,1,'A',4,20155,'','91',7,21167,21167,49),('VA',1484,1,1,1,1,'A',4,13860,'','305',34,15344,15344,51),('VT',321,1,1,1,1,'A',4,4174,'','45',16,4495,4495,50),('WA',5634,1,1,1,1,'A',4,60566,'','254',224,66200,66200,53),('WI',1550,1,1,1,1,'A',4,18819,'','398',24,20369,20369,55),('WV',191,1,1,1,0,'B',3,4384,'','1',1,4575,4575,54),('WY',130,1,1,1,1,'A',4,2218,'','18',0,2348,2348,56),('PR',286,1,1,1,1,'A',4,1409,'897','',11,2592,1695,72);
/*!40000 ALTER TABLE `states_covidcases` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2020-04-09 16:54:13
