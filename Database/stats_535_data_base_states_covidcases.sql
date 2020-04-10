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

-- Dump completed on 2020-04-09 16:55:12
